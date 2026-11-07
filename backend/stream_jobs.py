"""Atomic, project-scoped media spools on the native temporary volume."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from backend.storage import atomic_json, read_json, safe_id, sha256


def stream_base(root):
    identity=hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:24]
    base=Path(tempfile.gettempdir())/'digitalhuman-photo'/identity
    if base.resolve()!=base.absolute():raise ValueError('流式目录包含目录链接')
    base.mkdir(parents=True,exist_ok=True,mode=0o700)
    return base


def stream_folder(root,session_id,generation,index):
    if any(type(value) is not int or value<0 for value in (generation,index)):
        raise ValueError('回答或句子编号无效')
    base=stream_base(root)
    folder=base/safe_id(session_id)/f'{generation}_{index}'
    if folder.resolve()!=folder:raise ValueError('流式目录包含目录链接')
    return folder


def check_space(root,temporary,reserve_gib=10):
    for path in (Path(root),Path(root)/'data',Path(temporary)):
        while not path.exists():path=path.parent
        if shutil.disk_usage(path).free<reserve_gib*1024**3:
            raise OSError(f'流式生成需要保留至少{reserve_gib} GiB磁盘空间')


class AudioSpool:
    def __init__(self,folder,rate=None):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'audio.json'
        if rate is not None:
            if type(rate) is not int or rate<=0:raise ValueError('采样率无效')
            if self.path.exists():raise ValueError('声音流已经存在')
            atomic_json(self.path,dict(rate=rate,chunks=0,samples=0,finished=False,error=None))

    def state(self):return read_json(self.path)

    def append(self,pcm):
        state=self.state()
        pcm=np.asarray(pcm,dtype=np.float32)
        if state['finished'] or pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():
            raise ValueError('声音分块无效或声音流已结束')
        index=state['chunks'];path=self.folder/f'audio-{index}.npy'
        temporary=path.with_name('.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('wb') as output:np.save(output,pcm,allow_pickle=False)
            os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)
        event=dict(type='chunk',chunk_index=index,sample_start=state['samples'],
                   samples=len(pcm),rate=state['rate'],sha256=sha256(path))
        atomic_json(self.folder/f'audio-{index}.json',event)
        atomic_json(self.path,{**state,'chunks':index+1,'samples':state['samples']+len(pcm)})
        return event

    def read(self,event):
        index=event.get('chunk_index')
        if type(index) is not int or index<0:raise ValueError('声音块编号无效')
        path=self.folder/f'audio-{index}.npy'
        expected=read_json(self.folder/f'audio-{index}.json')
        if any(event.get(key)!=value for key,value in expected.items()):raise ValueError('声音块范围不匹配')
        if sha256(path)!=expected['sha256']:raise ValueError('声音块校验失败')
        with path.open('rb') as source:pcm=np.load(source,allow_pickle=False)
        if pcm.dtype!=np.float32 or pcm.shape!=(event['samples'],) or not np.isfinite(pcm).all():
            raise ValueError('声音块数据无效')
        return pcm

    def finish(self,*,error=None,report=None):
        state=self.state()
        atomic_json(self.path,{**state,'finished':True,'error':error,'report':report})


def video_chunk_path(folder,index):
    if type(index) is not int or index<0:raise ValueError('画面块编号无效')
    return Path(folder)/f'video-{index}.mkv'


async def iter_worker_events(base_url,endpoint,request,*,job_path=None):
    import asyncio
    import httpx
    client=await asyncio.to_thread(httpx.AsyncClient,timeout=600,trust_env=False)
    started=completed=False
    try:
        async with client:
            async with client.stream('POST',base_url+endpoint,json=request) as response:
                if response.is_error:
                    await response.aread()
                    try:
                        detail=response.json()
                        if detail.get('code')=='stream_memory':
                            from backend.errors import StreamMemoryUnavailable
                            raise StreamMemoryUnavailable(detail.get('detail','显存余量不足'))
                    except (ValueError,TypeError):pass
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:continue
                    if len(line)>65536:raise ValueError('流式事件超过长度上限')
                    event=json.loads(line)
                    if (not isinstance(event,dict) or type(event.get('generation')) is not int
                            or event.get('generation')!=request['generation']
                            or event.get('sentence_index')!=request['index']):
                        raise ValueError('流式事件不属于当前回答')
                    kind=event.get('type')
                    if completed:raise ValueError('流结束后仍收到事件')
                    if kind=='error':
                        if event.get('code')=='stream_memory':
                            from backend.errors import StreamMemoryUnavailable
                            raise StreamMemoryUnavailable(event.get('text','显存余量不足'))
                        raise RuntimeError(event.get('text','模型流式生成失败'))
                    if kind=='start':
                        if started:raise ValueError('流重复开始')
                        started=True
                    elif not started or kind not in {'chunk','complete'}:raise ValueError('流式事件顺序无效')
                    if kind=='complete':completed=True
                    yield event
        if not completed:raise RuntimeError('模型流在完成前断开')
    finally:
        if job_path is not None:
            from backend.stream_worker import wait_worker_done
            await wait_worker_done(job_path)

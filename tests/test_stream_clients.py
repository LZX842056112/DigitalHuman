import asyncio
import json
import threading

import httpx
import numpy as np
import pytest

from backend.stream_jobs import AudioSpool,stream_folder


def transport(monkeypatch,events):
    original=httpx.AsyncClient
    def respond(request):
        return httpx.Response(200,text='\n'.join(json.dumps(event) for event in events)+'\n')
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(**kwargs,transport=httpx.MockTransport(respond)))


def test_voice_client_reads_contiguous_spool_and_requires_complete(tmp_path,monkeypatch):
    from backend.voice.client import VoiceClient
    spool=AudioSpool(stream_folder(tmp_path,'session',1,0),24000)
    chunk=spool.append(np.ones(2400,np.float32)*.1);spool.finish(report={})
    events=[dict(type='start',rate=24000),chunk,dict(type='complete',samples=2400,chunks=1,report={})]
    events=[{**event,'generation':1,'sentence_index':0} for event in events]
    transport(monkeypatch,events)
    async def collect():
        return [event async for event in VoiceClient().stream('你好',tmp_path/'person',{'confirmed':True},
            root=tmp_path,session_id='session',generation=1,index=0,token='x'*64)]
    result=asyncio.run(collect())
    np.testing.assert_array_equal(result[1]['pcm'],np.full(2400,.1,np.float32))
    assert result[-1]['type']=='complete'


def test_spool_setup_runs_outside_the_media_event_loop(tmp_path,monkeypatch):
    from backend.voice.client import VoiceClient
    from backend import stream_jobs
    spool=AudioSpool(stream_folder(tmp_path,'session',1,0),24000)
    chunk=spool.append(np.full(2400,.1,np.float32));spool.finish()
    events=[dict(type='start',rate=24000),chunk,dict(type='complete',samples=2400,chunks=1,report={})]
    transport(monkeypatch,[{**event,'generation':1,'sentence_index':0} for event in events])
    original=stream_jobs.stream_folder;threads=[];main_thread=threading.get_ident()
    def folder(*args):threads.append(threading.get_ident());return original(*args)
    monkeypatch.setattr(stream_jobs,'stream_folder',folder)
    async def collect():
        return [event async for event in VoiceClient().stream('你好',tmp_path/'person',{'confirmed':True},
            root=tmp_path,session_id='session',generation=1,index=0,token='x'*64)]
    asyncio.run(collect())
    assert threads and all(thread!=main_thread for thread in threads)


@pytest.mark.parametrize('fault',['duplicate','wrong_generation','missing_complete'])
def test_voice_client_rejects_invalid_or_truncated_stream(tmp_path,monkeypatch,fault):
    from backend.voice.client import VoiceClient
    spool=AudioSpool(stream_folder(tmp_path,'session',1,0),24000)
    chunk=spool.append(np.ones(2400,np.float32)*.1);spool.finish()
    events=[dict(type='start',rate=24000),chunk,dict(type='complete',samples=2400,chunks=1,report={})]
    if fault=='duplicate':events.insert(2,chunk)
    if fault=='missing_complete':events.pop()
    events=[{**event,'generation':0 if fault=='wrong_generation' else 1,'sentence_index':0} for event in events]
    transport(monkeypatch,events)
    async def collect():
        return [event async for event in VoiceClient().stream('你好',tmp_path/'person',{'confirmed':True},
            root=tmp_path,session_id='session',generation=1,index=0,token='x'*64)]
    with pytest.raises((ValueError,RuntimeError)):asyncio.run(collect())


@pytest.mark.parametrize('kind',['voice','portrait'])
def test_protocol_failure_waits_for_inner_worker_drain(tmp_path,monkeypatch,kind):
    from backend.voice.client import VoiceClient
    from backend.portrait.client import PortraitClient
    async def check():
        closing=asyncio.Event();release=asyncio.Event()
        async def events(*args,**kwargs):
            try:yield dict(type='start',rate=-1,fps=-1)
            finally:closing.set();await release.wait()
        monkeypatch.setattr('backend.stream_jobs.iter_worker_events',events)
        async def collect():
            iterator=(VoiceClient().stream('你好',tmp_path/'person',{'confirmed':True},root=tmp_path,
                session_id='session',generation=1,index=0,token='x'*64) if kind=='voice' else
                PortraitClient().stream('person','session',1,0,'neutral',root=tmp_path,token='x'*64))
            return [event async for event in iterator]
        task=asyncio.create_task(collect())
        try:
            await asyncio.wait_for(closing.wait(),1)
            assert not task.done(),'协议错误不能先于原生生产者清理完成'
        finally:
            release.set()
            result=await asyncio.gather(task,return_exceptions=True)
            assert isinstance(result[0],ValueError)
    asyncio.run(check())

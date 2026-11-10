import asyncio
import httpx
from contextlib import aclosing
from backend.emotion import normalize_emotion
from backend.style import normalize_expression


class PortraitClient:
    def __init__(self,base_url='http://127.0.0.1:9881'):
        self.base_url=base_url

    async def _post(self,route,body):
        client=await asyncio.to_thread(httpx.AsyncClient,timeout=600,trust_env=False)
        async with client:
            response=await client.post(self.base_url+route,json=body)
            if response.is_error:raise RuntimeError('照片服务：'+response.text[:500])
            return response.json()

    async def prepare(self,avatar_id,variant_id=None):
        body={'avatar_id':avatar_id}
        if variant_id:body['variant_id']=variant_id
        return await self._post('/prepare',body)

    async def animate(self,avatar_id,session_id,generation,index,emotion,expression='natural'):
        return await self._post('/render',{'avatar_id':avatar_id,'session_id':session_id,
            'generation':generation,'index':index,'emotion':normalize_emotion(emotion),
            'expression':normalize_expression(expression)})

    async def stream(self,avatar_id,session_id,generation,index,emotion,expression='natural',*,root,token):
        from backend.stream_jobs import iter_worker_events,stream_folder,video_chunk_path
        request=dict(avatar_id=avatar_id,session_id=session_id,generation=generation,index=index,
                     emotion=normalize_emotion(emotion),expression=normalize_expression(expression),
                     token=token)
        chunks=frames=0
        folder=await asyncio.to_thread(stream_folder,root,session_id,generation,index)
        async with aclosing(iter_worker_events(self.base_url,'/render/stream',request,
                job_path=folder/'portrait-job.json')) as events:
            async for event in events:
                if event['type']=='start':
                    if event.get('fps')!=25:raise ValueError('画面流帧率无效')
                elif event['type']=='chunk':
                    if (event.get('chunk_index')!=chunks or event.get('frame_start')!=frames
                            or type(event.get('frames')) is not int or not 1<=event['frames']<=25):
                        raise ValueError('画面分块重复、缺失或乱序')
                    path=video_chunk_path(folder,chunks)
                    if not await asyncio.to_thread(path.is_file):raise ValueError('画面分块尚未发布')
                    event={**event,'video':path};chunks+=1;frames+=event['frames']
                elif event['type']=='complete':
                    if not chunks or event.get('report',{}).get('frames')!=frames:raise ValueError('画面流未完整结束')
                yield event

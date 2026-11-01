import asyncio
import httpx
from backend.emotion import normalize_emotion


class PortraitClient:
    def __init__(self,base_url='http://127.0.0.1:9881'):
        self.base_url=base_url

    async def _post(self,route,body):
        client=await asyncio.to_thread(httpx.AsyncClient,timeout=600,trust_env=False)
        async with client:
            response=await client.post(self.base_url+route,json=body)
            if response.is_error:raise RuntimeError('照片服务：'+response.text[:500])
            return response.json()

    async def prepare(self,avatar_id):
        return await self._post('/prepare',{'avatar_id':avatar_id})

    async def animate(self,avatar_id,session_id,generation,index,emotion):
        return await self._post('/render',{'avatar_id':avatar_id,'session_id':session_id,
            'generation':generation,'index':index,'emotion':normalize_emotion(emotion)})

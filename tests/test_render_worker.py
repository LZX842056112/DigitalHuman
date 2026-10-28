import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from backend.avatar.expressions import ExpressionMap
from backend.transport.session import Session


async def test_rendering_can_continue_while_shared_codec_pool_is_busy(tmp_path,monkeypatch):
    class Renderer:
        default_camera={}
        def render(self,*args): return np.zeros((8,8,3),np.uint8)
    monkeypatch.setattr(ExpressionMap,'load',lambda *_:None)
    session=Session(tmp_path,{'id':'avatar'},Renderer(),{})
    shared=ThreadPoolExecutor(1);asyncio.get_running_loop().set_default_executor(shared)
    entered=threading.Event();release=threading.Event()
    def codec(): entered.set();release.wait(3)
    occupied=shared.submit(codec);pending=None
    try:
        assert entered.wait(1)
        pending=asyncio.create_task(session.render_frame(np.zeros(100),np.zeros(3)))
        image=await asyncio.wait_for(asyncio.shield(pending),.5)
        assert image.shape==(8,8,3)
    finally:
        release.set();occupied.result(3)
        if pending is not None: await asyncio.gather(pending,return_exceptions=True)
        await session.close()
        shared.shutdown(wait=True)

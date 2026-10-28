import asyncio
from types import SimpleNamespace
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError
from backend.transport.webrtc import AvatarVideoTrack


class RenderSession:
    def __init__(self,block_second=False,fail=False):
        self.timeline=SimpleNamespace(expression=lambda:(np.zeros(100),np.zeros(3)))
        self.closed=False;self.frames=0;self.render_seconds=0.;self.calls=0
        self.video_queue_depth=0;self.video_frames_dropped=0;self.errors=[]
        self.second_started=asyncio.Event();self.sixth_started=asyncio.Event()
        self.gate=asyncio.Event();self.block_second=block_second;self.fail=fail

    async def render_frame(self,*args):
        self.calls+=1
        if self.fail: raise RuntimeError('test render failure')
        if self.calls==2:
            self.second_started.set()
            if self.block_second: await self.gate.wait()
        if self.calls>=6: self.sixth_started.set()
        return np.full((8,8,3),self.calls%256,np.uint8)

    def emit(self,kind,**payload): self.errors.append((kind,payload))


async def close_track(track):
    track.stop()
    task=getattr(track,'_producer',None)
    if task is not None: await asyncio.gather(task,return_exceptions=True)


async def test_next_render_starts_while_consumer_is_encoding_previous_frame():
    session=RenderSession(block_second=True);track=AvatarVideoTrack(session,200)
    try:
        first=await track.recv()
        assert first.to_ndarray(format='rgb24')[0,0,0]==1
        await asyncio.wait_for(session.second_started.wait(),1)
    finally:
        await close_track(track)


async def test_slow_consumer_gets_latest_real_frame_and_queue_stays_bounded():
    session=RenderSession();track=AvatarVideoTrack(session,200)
    try:
        first=await track.recv()
        await asyncio.wait_for(session.sixth_started.wait(),1)
        assert session.video_queue_depth<=1 and session.video_frames_dropped>=3
        latest=await track.recv()
        assert latest.to_ndarray(format='rgb24')[0,0,0]>=5
        assert latest.pts>first.pts
    finally:
        await close_track(track)


async def test_render_failure_wakes_receiver_and_preserves_original_error():
    session=RenderSession(fail=True);track=AvatarVideoTrack(session,15)
    try:
        with pytest.raises(MediaStreamError) as error:
            await asyncio.wait_for(track.recv(),1)
        assert isinstance(error.value.__cause__,RuntimeError)
        assert session.errors and 'test render failure' in session.errors[0][1]['text']
    finally:
        await close_track(track)


async def test_stopping_video_keeps_reservation_until_inflight_render_finishes(tmp_path):
    import threading
    import time
    from backend.transport.session import Session
    entered=threading.Event();release=threading.Event();released=[]
    class BlockingRenderer:
        def render(self,*args):
            entered.set();release.wait(3)
            return np.zeros((8,8,3),np.uint8)
    session=Session.__new__(Session)
    session._close_lock=asyncio.Lock();session._render_lock=asyncio.Lock();session._render_task=None
    session.closed=False;session.renderer=BlockingRenderer();session.action='idle';session.camera={}
    session.position=np.zeros(3);session.started=time.monotonic();session.peer=None;session.channel=None;session.task=None
    session.timeline=SimpleNamespace(expression=lambda:(np.zeros(100),np.zeros(3)),interrupt=lambda:None)
    session.frames=0;session.render_seconds=0.;session.gpu_lease=SimpleNamespace(release=lambda:released.append(True))
    track=AvatarVideoTrack(session,15);receive=asyncio.create_task(track.recv());closing=None
    try:
        assert await asyncio.to_thread(entered.wait,1)
        track.stop()
        with pytest.raises(MediaStreamError): await asyncio.wait_for(receive,1)
        closing=asyncio.create_task(session.close())
        await asyncio.sleep(.03)
        assert not released and not closing.done()
    finally:
        release.set()
        if closing is not None: await closing
        await close_track(track)
    assert released==[True] and session.renderer is None and session._render_task is None


async def test_session_close_joins_video_producer_before_releasing_gpu():
    import time
    import weakref
    import gc
    from backend.transport.session import Session
    class Renderer:
        def render(self,*args): return np.zeros((8,8,3),np.uint8)
    session=Session.__new__(Session)
    session._close_lock=asyncio.Lock();session._render_lock=asyncio.Lock();session._render_task=None
    session.closed=False;session.renderer=Renderer();session.action='idle';session.camera={}
    session.position=np.zeros(3);session.started=time.monotonic();session.channel=None;session.task=None
    session.timeline=SimpleNamespace(expression=lambda:(np.zeros(100),np.zeros(3)),interrupt=lambda:None)
    session.frames=0;session.render_seconds=0.
    reference=weakref.ref(session.renderer)
    track=AvatarVideoTrack(session,15);session.video_track=track
    class Peer:
        async def close(self): track.stop()
    session.peer=Peer();released=[]
    session.gpu_lease=SimpleNamespace(release=lambda:released.append(track._producer is None or track._producer.done()))
    try:
        await track.recv()
        await session.close()
        await asyncio.sleep(.05)
        gc.collect()
        assert reference() is None,'A completed video producer retained its renderer'
    finally:
        await close_track(track)
    assert released==[True],'GPU reservation was released while its video producer was still unwinding'


async def test_one_late_render_does_not_shift_every_later_frame_deadline(monkeypatch):
    from backend.transport import webrtc
    clock=SimpleNamespace(now=0.)
    async def sleep(seconds):
        clock.now+=max(0,seconds)
        await asyncio.sleep(0)
    monkeypatch.setattr(webrtc,'time',SimpleNamespace(monotonic=lambda:clock.now))
    monkeypatch.setattr(webrtc,'asyncio',SimpleNamespace(Queue=asyncio.Queue,create_task=asyncio.create_task,
        sleep=sleep,CancelledError=asyncio.CancelledError,current_task=asyncio.current_task,gather=asyncio.gather))
    class TimedSession(RenderSession):
        async def render_frame(self,*args):
            if self.calls>=3: await self.gate.wait()
            clock.now+=.09 if self.calls==0 else .04
            return await super().render_frame(*args)
    session=TimedSession();track=AvatarVideoTrack(session,15)
    try:
        frames=[await track.recv() for _ in range(3)]
        times=[float(frame.pts*frame.time_base) for frame in frames]
        assert .089<=times[0]<=.091
        assert times[2]<.185,'A single90ms render permanently shifted the15FPS clock'
        assert times==sorted(set(times)),'Recovery must use real frames with increasing timestamps'
    finally:
        await close_track(track)

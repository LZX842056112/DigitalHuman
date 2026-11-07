import asyncio
import threading
from types import SimpleNamespace

import numpy as np

from test_photo_session import session,until,movie


def streaming_clients(instance,release):
    from backend.emotion import Reply
    from backend.stream_jobs import stream_folder
    entered=asyncio.Event();voice_done=asyncio.Event()
    async def answer(*args):return Reply('你好。','neutral')
    async def buffered(*args,**kwargs):raise AssertionError('流式配置仍走整句路线')
    async def voice(*args,**kwargs):
        entered.set();yield dict(type='start',rate=24000)
        for index in range(3):
            yield dict(type='chunk',pcm=np.full(24000,.1,np.float32),rate=24000,
                chunk_index=index,sample_start=index*24000,samples=24000)
        await release.wait()
        yield dict(type='chunk',pcm=np.full(24000,.1,np.float32),rate=24000,
            chunk_index=3,sample_start=72000,samples=24000)
        voice_done.set();yield dict(type='complete',samples=96000,chunks=4,report={})
    async def portrait(avatar,sid,generation,index,emotion,**kwargs):
        folder=stream_folder(instance.root,sid,generation,index);folder.mkdir(parents=True,exist_ok=True)
        yield dict(type='start',fps=25)
        for chunk in range(4):
            if chunk==3:await release.wait()
            path=await asyncio.to_thread(movie,folder/f'video-{chunk}.mkv',60+chunk)
            yield dict(type='chunk',chunk_index=chunk,frame_start=chunk*25,frames=25,video=path)
        yield dict(type='complete',report={'frames':100})
    instance.dialogue=SimpleNamespace(answer=answer)
    instance.voice=SimpleNamespace(stream=voice,synthesize=buffered)
    instance.portrait=SimpleNamespace(stream=portrait)
    return entered,voice_done


def test_enabled_session_plays_paired_prefix_before_voice_finishes(tmp_path):
    async def check():
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();entered,voice_done=streaming_clients(instance,release)
        instance.message('{"type":"chat_text","text":"你好"}')
        try:
            await until(lambda:entered.is_set() or instance.task.done())
            assert entered.is_set(),events
            await until(lambda:instance._turn_streams and instance._turn_streams[0].frames>=75)
            pcm,_=instance.timeline.read_audio(960)
            assert np.any(pcm) and not voice_done.is_set()
            assert instance.metrics['last_turn_timing']['playback_start_ms'] is not None
            release.set()
            await until(lambda:instance._turn_streams[0].finished)
            for _ in range(250):instance.timeline.read_audio(960)
            await until(lambda:instance.task.done())
            assert any(event['type']=='turn_complete' for event in events)
            assert not any(event['type']=='error' for event in events)
            assert instance.metrics['stream_underruns']==0
        finally:
            release.set();await instance.close()
    asyncio.run(check())


def test_timeline_failure_waits_for_client_generator_cleanup(tmp_path):
    async def check():
        from backend.emotion import Reply
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        closing=asyncio.Event();release=asyncio.Event()
        async def answer(*args):return Reply('你好。','neutral')
        async def voice(*args,**kwargs):
            try:
                yield dict(type='start',rate=24000)
                yield dict(type='chunk',pcm=np.full(2400,.1,np.float32))
            finally:closing.set();await release.wait()
        async def portrait(*args,**kwargs):
            if False:yield {}
        def invalid(*args):raise ValueError('timeline validation failed')
        instance.dialogue=SimpleNamespace(answer=answer)
        instance.voice=SimpleNamespace(stream=voice);instance.portrait=SimpleNamespace(stream=portrait)
        instance.timeline.update_stream=invalid
        instance.message('{"type":"chat_text","text":"你好"}')
        try:
            await asyncio.wait_for(closing.wait(),2)
            await asyncio.wait([instance.task],timeout=.1)
            assert not instance.task.done(),'时间轴错误不能丢失客户端清理任务'
        finally:
            release.set();await instance.close()
    asyncio.run(check())


def test_late_completion_cannot_hide_extra_frames_after_media_has_closed(tmp_path):
    async def check():
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();allow_complete=asyncio.Event();streaming_clients(instance,release)
        original=instance.portrait.stream
        async def portrait(*args,**kwargs):
            async for event in original(*args,**kwargs):
                if event['type']=='complete':
                    await allow_complete.wait();event={**event,'report':{'frames':101}}
                yield event
        instance.portrait.stream=portrait
        instance.message('{"type":"chat_text","text":"你好"}');release.set()
        try:
            await until(lambda:instance._turn_streams and instance._turn_streams[0].audio_finished
                        and instance._turn_streams[0].frames==100)
            clip=instance._turn_streams[0]
            for _ in range(201):instance.timeline.read_audio(960)
            assert clip.closed and not clip.video_finished
            allow_complete.set();await until(lambda:instance.task.done())
            assert any(event['type']=='error' for event in events)
            assert not any(event['type']=='turn_complete' for event in events)
        finally:allow_complete.set();release.set();await instance.close()
    asyncio.run(check())


def test_close_waits_for_stream_drain_and_releases_its_gpu_ownership(tmp_path):
    async def check():
        from backend.gpu import GpuLease
        from backend.stream_jobs import stream_folder
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();entered,_=streaming_clients(instance,release)
        instance.message('{"type":"chat_text","text":"你好"}')
        close=None
        try:
            await until(lambda:entered.is_set() or instance.task.done())
            assert entered.is_set(),events
            await until(lambda:instance._turn_streams and instance._turn_streams[0].frames>=25)
            generation=instance.generation
            close=asyncio.create_task(instance.close());await until(lambda:instance.closed)
            assert not close.done()
            release.set();await asyncio.wait_for(close,3)
            folder=stream_folder(tmp_path,instance.id,generation,0)
            assert not folder.exists()
            assert not instance._model_tasks
            GpuLease(tmp_path).acquire().release()
        finally:
            release.set()
            if close is not None:await close
            else:await instance.close()
    asyncio.run(check())


def test_worker_failure_invalidates_generation_before_other_worker_drains(tmp_path):
    async def check():
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();entered,_=streaming_clients(instance,release)
        original=instance.portrait.stream;allow_failure=asyncio.Event()
        async def broken(*args,**kwargs):
            async for event in original(*args,**kwargs):
                yield event
                if event['type']=='chunk':
                    await allow_failure.wait()
                    raise RuntimeError('画面生成失败测试')
        instance.portrait.stream=broken
        instance.message('{"type":"chat_text","text":"你好"}')
        try:
            await until(lambda:entered.is_set() or instance.task.done())
            generation=instance.generation
            allow_failure.set()
            await until(lambda:any(event['type']=='error' for event in events))
            assert instance.generation!=generation
            assert instance._model_tasks and not release.is_set()
            assert not any(event['type']=='turn_complete' for event in events)
        finally:allow_failure.set();release.set();await instance.close()
    asyncio.run(check())


def test_cancel_during_gpu_claim_keeps_late_lease_owned_until_cleanup(tmp_path,monkeypatch):
    async def check():
        from backend.gpu import SessionGpuLease,GpuLease
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();streaming_clients(instance,release)
        entered=threading.Event();allow=threading.Event();finished=threading.Event();leases=[]
        original=SessionGpuLease.acquire
        def blocked(lease):
            entered.set();allow.wait(3)
            try:
                result=original(lease);leases.append(result);return result
            finally:finished.set()
        monkeypatch.setattr(SessionGpuLease,'acquire',blocked)
        instance.message('{"type":"chat_text","text":"你好"}')
        closing=None
        try:
            await until(entered.is_set)
            instance.interrupt();closing=asyncio.create_task(instance.close())
            await asyncio.sleep(.03)
            assert not closing.done()
            allow.set();await asyncio.wait_for(closing,3)
            GpuLease(tmp_path).acquire().release()
        finally:
            allow.set();release.set()
            await asyncio.to_thread(finished.wait,3)
            if closing:await closing
            else:await instance.close()
            for lease in leases:lease.release()
    asyncio.run(check())


def test_memory_failure_before_playback_retries_serial_without_repeating_transcript(tmp_path,monkeypatch):
    async def check():
        from backend.errors import StreamMemoryUnavailable
        monkeypatch.setattr('backend.gpu.stream_admission',lambda root:(True,None))
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();entered,_=streaming_clients(instance,release)
        original=instance.portrait.stream;attempts=[]
        async def limited(*args,**kwargs):
            attempts.append(1)
            if len(attempts)==1:raise StreamMemoryUnavailable('显存余量不足')
            async for event in original(*args,**kwargs):yield event
        instance.portrait.stream=limited
        instance.message('{"type":"chat_text","text":"你好"}')
        try:
            await until(entered.is_set);release.set()
            await until(lambda:len(attempts)==2 and instance._turn_streams[-1].finished)
            for _ in range(250):instance.timeline.read_audio(960)
            await until(lambda:instance.task.done())
            assert instance.metrics['delivery_mode']=='streaming_serial'
            assert instance.metrics['last_turn_timing']['stream_retries']==1
            assert len([event for event in events if event['type']=='transcript'])==2
            assert not any(event['type']=='error' for event in events)
            assert any(event['type']=='turn_complete' for event in events)
        finally:release.set();await instance.close()
    asyncio.run(check())


def test_peer_close_error_does_not_abandon_stream_or_gpu_lease(tmp_path):
    async def check():
        import pytest
        from backend.gpu import GpuLease
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();entered,_=streaming_clients(instance,release)
        async def broken_close():raise RuntimeError('连接关闭失败测试')
        instance.peer=SimpleNamespace(close=broken_close)
        instance.message('{"type":"chat_text","text":"你好"}')
        closing=None
        try:
            await until(entered.is_set)
            closing=asyncio.create_task(instance.close());await until(lambda:instance.closed)
            await asyncio.sleep(.02);assert not closing.done()
            release.set()
            with pytest.raises(RuntimeError,match='连接关闭失败测试'):await closing
            assert not instance._model_tasks and instance.timeline._idle_video is None
            GpuLease(tmp_path).acquire().release()
        finally:
            release.set()
            if closing is not None:await asyncio.gather(closing,return_exceptions=True)
            await asyncio.gather(*list(instance._model_tasks),return_exceptions=True)
            instance.timeline.close();instance._cleanup_streams()
            if instance._stream_lease is not None:instance._stream_lease.release()
    asyncio.run(check())


def test_memory_failure_after_playback_stops_without_automatic_replay(tmp_path,monkeypatch):
    async def check():
        from backend.errors import StreamMemoryUnavailable
        monkeypatch.setattr('backend.gpu.stream_admission',lambda root:(True,None))
        instance,events=session(tmp_path);instance.config={'runtime':{'photo_streaming':True}}
        release=asyncio.Event();streaming_clients(instance,release)
        original=instance.portrait.stream;fail=asyncio.Event();attempts=[]
        async def limited(*args,**kwargs):
            attempts.append(1)
            async for event in original(*args,**kwargs):
                yield event
                if event['type']=='chunk' and event['chunk_index']==2:
                    await fail.wait();raise StreamMemoryUnavailable('显存余量不足')
        instance.portrait.stream=limited
        instance.message('{"type":"chat_text","text":"你好"}')
        try:
            await until(lambda:instance._turn_streams and instance._turn_streams[0].frames>=75)
            assert np.any(instance.timeline.read_audio(960)[0])
            fail.set();await until(lambda:any(event['type']=='error' for event in events))
            release.set();await until(lambda:instance.task is None or instance.task.done())
            assert len(attempts)==1
            assert instance.metrics['last_turn_timing']['status']=='failed'
            assert not any(event['type']=='turn_complete' for event in events)
            assert len([event for event in events if event['type']=='transcript'])==2
        finally:fail.set();release.set();await instance.close()
    asyncio.run(check())

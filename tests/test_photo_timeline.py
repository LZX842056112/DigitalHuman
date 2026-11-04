import asyncio
import time
from types import SimpleNamespace

import av
import numpy as np
import pytest


def movie(path,frames=25):
    with av.open(str(path),'w') as container:
        stream=container.add_stream('ffv1',rate=25);stream.width=16;stream.height=16;stream.pix_fmt='bgr0'
        for index in range(frames):
            frame=av.VideoFrame.from_ndarray(np.full((16,16,3),index,np.uint8),format='rgb24')
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return path


def test_photo_video_follows_consumed_audio_samples(tmp_path):
    from backend.transport.photo_timeline import PhotoTimeline, SentenceClip
    timeline=PhotoTimeline(np.full((16,16,3),200,np.uint8))
    clip=SentenceClip(np.ones(16000,np.float32)*.1,16000,movie(tmp_path/'test.mkv'))
    token=timeline.begin_turn();timeline.append(token,clip)
    pcm,pts=timeline.read_audio(1920)
    assert pts == 0 and len(pcm) == 1920
    assert abs(float(pcm.mean())-.1) < 1e-3
    assert timeline.image()[0,0,0] == 1
    timeline.read_audio(1920)
    assert timeline.image()[0,0,0] == 2
    timeline.close()


def test_interrupt_closes_clips_and_rejects_stale_generation(tmp_path):
    from backend.transport.photo_timeline import PhotoTimeline, SentenceClip
    idle=np.full((16,16,3),200,np.uint8);timeline=PhotoTimeline(idle)
    token=timeline.begin_turn()
    clip=SentenceClip(np.ones(48000,np.float32),48000,movie(tmp_path/'old.mkv'),delete=True)
    timeline.append(token,clip);timeline.read_audio(960);timeline.interrupt()
    assert not (tmp_path/'old.mkv').exists()
    assert not timeline.busy and timeline.queue_depth == 0
    assert timeline.read_audio(960)[0].max() == 0
    np.testing.assert_array_equal(timeline.image(),idle)
    late=SentenceClip(np.ones(48000,np.float32),48000,movie(tmp_path/'late.mkv'),delete=True)
    assert not timeline.append(token,late)
    assert not (tmp_path/'late.mkv').exists()
    next_token=timeline.begin_turn()
    assert timeline.append(next_token,SentenceClip(np.ones(48000,np.float32),48000,movie(tmp_path/'new.mkv')))
    assert timeline.read_audio(960)[0].mean() == 1
    timeline.close()


def test_only_two_sentences_can_wait_for_playback(tmp_path):
    from backend.transport.photo_timeline import PhotoTimeline, SentenceClip
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8));token=timeline.begin_turn()
    clips=[SentenceClip(np.zeros(48000,np.float32),48000,movie(tmp_path/f'{i}.mkv')) for i in range(3)]
    timeline.append(token,clips[0]);timeline.append(token,clips[1])
    with pytest.raises(BufferError):timeline.append(token,clips[2])
    assert timeline.queue_depth == 2
    clips[2].close();timeline.close()


def test_idle_animation_plays_without_spoken_audio(tmp_path,monkeypatch):
    from backend.transport import photo_timeline
    clock=[0.]
    monkeypatch.setattr(photo_timeline.clock,'monotonic',lambda:clock[0])
    timeline=photo_timeline.PhotoTimeline(np.zeros((16,16,3),np.uint8),idle_video=movie(tmp_path/'idle.mkv'))
    clock[0]=.08
    assert timeline.image()[0,0,0] == 2
    clock[0]=1.04
    assert timeline.image()[0,0,0] == 1
    assert timeline.read_audio(960)[0].max() == 0
    timeline.close()


@pytest.mark.asyncio
async def test_idle_phase_jump_does_not_block_audio_packet_on_decoder_lock(tmp_path,monkeypatch):
    from backend.transport import photo_timeline
    from backend.transport.photo_webrtc import PhotoAudioTrack
    path=movie(tmp_path/'idle.mkv')
    original_open=av.open

    class DelayedFrame:
        def __init__(self,frame):self.frame=frame
        def __getattr__(self,key):return getattr(self.frame,key)
        def to_ndarray(self,*args,**kwargs):
            time.sleep(.008)
            return self.frame.to_ndarray(*args,**kwargs)

    class DelayedContainer:
        def __init__(self,container):self.container=container
        def __getattr__(self,key):return getattr(self.container,key)
        def __enter__(self):return self
        def __exit__(self,*args):self.container.close()
        def decode(self,*args,**kwargs):
            for frame in self.container.decode(*args,**kwargs):yield DelayedFrame(frame)

    monkeypatch.setattr(av,'open',lambda *args,**kwargs:DelayedContainer(original_open(*args,**kwargs)))
    # Keep asyncio's monotonic clock real. Only the timeline's idle origin is
    # changed, as happens after a long interval of spoken-video playback.
    timeline=photo_timeline.PhotoTimeline(np.zeros((16,16,3),np.uint8),idle_video=path)
    timeline._idle_started=photo_timeline.clock.monotonic()-.88
    track=PhotoAudioTrack(SimpleNamespace(timeline=timeline,closed=False))
    decoding=asyncio.create_task(asyncio.to_thread(timeline.video_frame))
    try:
        await asyncio.sleep(.02)
        started=time.monotonic()
        packet=await track.recv()
        elapsed=time.monotonic()-started
        assert packet.samples==960
        assert elapsed<.04,f'Idle decoder blocked the audio packet for {elapsed:.3f}s'
        await decoding
    finally:
        await asyncio.gather(decoding,return_exceptions=True)
        track.stop();timeline.close()


def test_idle_cache_preserves_wall_phase_after_speech_and_interrupt(tmp_path,monkeypatch):
    from backend.transport import photo_timeline
    clock=[0.]
    monkeypatch.setattr(photo_timeline.clock,'monotonic',lambda:clock[0])
    timeline=photo_timeline.PhotoTimeline(np.zeros((16,16,3),np.uint8),idle_video=movie(tmp_path/'idle.mkv'))
    try:
        for seconds in [.88,21.08,21.96,22.0]:
            clock[0]=seconds
            expected=int((seconds%1)*25)
            assert timeline.image()[0,0,0]==expected
        timeline.interrupt();clock[0]=22.08
        assert timeline.image()[0,0,0]==int((22.08%1)*25)
    finally:timeline.close()


def test_idle_cache_rejects_more_than_one_hundred_frames(tmp_path):
    from backend.transport.photo_timeline import PhotoTimeline
    path=movie(tmp_path/'too-long.mkv',frames=101)
    with pytest.raises(ValueError,match='待机.*100'):
        PhotoTimeline(np.zeros((16,16,3),np.uint8),idle_video=path)


@pytest.mark.asyncio
@pytest.mark.parametrize('decode_error',[False,True])
async def test_slow_speech_decode_does_not_block_pcm_or_interrupt(tmp_path,decode_error):
    import threading
    from backend.transport.photo_timeline import PhotoTimeline, SentenceClip
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8))
    path=movie(tmp_path/'speech.mkv')
    clip=SentenceClip(np.ones(48000,np.float32)*.1,48000,path,delete=True)
    timeline.append(timeline.begin_turn(),clip);timeline.read_audio(960)
    entered=threading.Event();release=threading.Event();original=clip.video.at
    def delayed(seconds):
        entered.set()
        assert release.wait(3),'decoder release timed out'
        if decode_error:raise ValueError('decode failed')
        return original(seconds)
    clip.video.at=delayed
    decoding=asyncio.create_task(asyncio.to_thread(timeline.video_frame))
    watchdog=threading.Timer(.4,release.set)
    try:
        assert await asyncio.to_thread(entered.wait,2)
        watchdog.start()
        started=time.monotonic()
        pcm,pts=timeline.read_audio(960)
        timeline.interrupt()
        assert time.monotonic()-started<.08,'speech decoder blocked PCM or interrupt'
        assert pts==960 and pcm.mean()==pytest.approx(.1)
        assert not clip.video.closed and path.exists()
        release.set()
        result=await asyncio.gather(decoding,return_exceptions=True)
        if decode_error:assert isinstance(result[0],ValueError)
        else:assert result[0][1]==960
        assert clip.video.closed and not path.exists()
        assert not timeline._readers and not timeline._retired
    finally:
        release.set();watchdog.cancel()
        await asyncio.gather(decoding,return_exceptions=True)
        timeline.close()


@pytest.mark.asyncio
async def test_interrupt_defers_close_until_all_video_readers_finish(tmp_path):
    import threading
    from backend.transport.photo_timeline import PhotoTimeline, SentenceClip
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8))
    clip=SentenceClip(np.ones(48000,np.float32),48000,movie(tmp_path/'shared.mkv'),delete=True)
    timeline.append(timeline.begin_turn(),clip);timeline.read_audio(960)
    entered=threading.Event();release=threading.Event();original=clip.video.at
    def delayed(seconds):
        entered.set();assert release.wait(3)
        assert not clip.video.closed
        return original(seconds)
    clip.video.at=delayed
    readers=[asyncio.create_task(asyncio.to_thread(timeline.video_frame)) for _ in range(2)]
    try:
        assert await asyncio.to_thread(entered.wait,2)
        async def both_pinned():
            while timeline._readers.get(clip)!=2:await asyncio.sleep(.001)
        await asyncio.wait_for(both_pinned(),2)
        timeline.interrupt()
        assert not clip.video.closed
        release.set();await asyncio.gather(*readers)
        assert clip.video.closed and not clip.video.path.exists()
        assert not timeline._readers and not timeline._retired
    finally:
        release.set();await asyncio.gather(*readers,return_exceptions=True);timeline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('action',['advance','close'])
async def test_clip_transition_or_close_keeps_active_reader_and_original_pcm_snapshot(tmp_path,action):
    import threading
    from backend.transport.photo_timeline import PhotoTimeline,SentenceClip
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8))
    clip=SentenceClip(np.ones(960,np.float32),48000,movie(tmp_path/'short.mkv'),delete=True)
    timeline.append(timeline.begin_turn(),clip);timeline.read_audio(960)
    entered,release=threading.Event(),threading.Event()
    original_at,original_close=clip.video.at,clip.close
    closes=[]
    def delayed(seconds):
        entered.set();assert release.wait(3)
        assert not clip.video.closed
        return original_at(seconds)
    def close():
        closes.append(True);original_close()
    clip.video.at=delayed;clip.close=close
    decoding=asyncio.create_task(asyncio.to_thread(timeline.video_frame))
    try:
        assert await asyncio.to_thread(entered.wait,2)
        if action=='advance':timeline.read_audio(960)
        else:timeline.close()
        assert not closes and clip.video.path.exists()
        release.set()
        image,pts=await asyncio.wait_for(decoding,1)
        assert image.shape==(16,16,3) and pts==960
        assert len(closes)==1 and not clip.video.path.exists()
        assert not timeline._readers and not timeline._retired
    finally:
        release.set();await asyncio.gather(decoding,return_exceptions=True);timeline.close()

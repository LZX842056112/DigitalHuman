import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError

from backend.transport.photo_timeline import PhotoTimeline


def session():
    return SimpleNamespace(timeline=PhotoTimeline(np.zeros((16, 16, 3), np.uint8)),
        closed=False, generation=0, frames=0, render_seconds=0., video_queue_depth=0,
        video_frames_dropped=0, recording=False, recorded=[], recorded_samples=0,
        emit=lambda *args, **kwargs: None, metrics={})


@pytest.mark.asyncio
async def test_audio_uses_pcm_clock_and_paces_packets():
    from backend.transport.photo_webrtc import PhotoAudioTrack
    instance = session()
    track = PhotoAudioTrack(instance)
    start = time.monotonic()
    frames = [await track.recv() for _ in range(4)]
    assert [frame.pts for frame in frames] == [0, 960, 1920, 2880]
    assert time.monotonic() - start >= .05
    assert all(frame.samples == 960 and frame.sample_rate == 48000 for frame in frames)
    track.stop()
    with pytest.raises(MediaStreamError):
        await track.recv()
    instance.timeline.close()


@pytest.mark.asyncio
async def test_media_packets_keep_real_cadence_when_adjusted_clock_runs_slow(monkeypatch):
    from backend.transport import photo_webrtc
    raw=(lambda:time.clock_gettime(time.CLOCK_MONOTONIC_RAW)) if hasattr(time,'CLOCK_MONOTONIC_RAW') else time.perf_counter
    # Reproduce the measured WSL tick=9167 without changing the kernel or
    # asyncio's clock. Only the transport's adjusted-clock view runs slow.
    monkeypatch.setattr(photo_webrtc,'time',SimpleNamespace(monotonic=lambda:raw()*.9167))
    instance=session();track=photo_webrtc.PhotoAudioTrack(instance)
    try:
        start=raw()
        frames=[await track.recv() for _ in range(25)]
        elapsed=raw()-start
        assert [frame.pts for frame in frames]==list(range(0,25*960,960))
        assert abs(elapsed-.48)<.03,f'48kHz packet clock drifted: {elapsed:.4f}s for 0.48s PCM'
    finally:track.stop();instance.timeline.close()


@pytest.mark.asyncio
async def test_video_queue_drops_old_frames_and_stop_wakes_receiver():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    track = PhotoVideoTrack(instance, 25)
    first = await track.recv()
    for _ in range(6):
        instance.timeline.read_audio(1440)
        await asyncio.sleep(.03)
    assert instance.video_queue_depth == 1
    newest = await track.recv()
    assert newest.pts > first.pts
    assert instance.video_frames_dropped >= 1
    await track.aclose()
    with pytest.raises(MediaStreamError):
        await track.recv()
    assert instance.video_queue_depth == 0
    instance.timeline.close()


@pytest.mark.asyncio
async def test_brief_sender_stall_recovers_cadence_with_one_fresh_pending_frame():
    from backend import clock
    from backend.transport.photo_webrtc import PhotoAudioTrack, PhotoVideoTrack
    instance = session()
    audio, video = PhotoAudioTrack(instance), PhotoVideoTrack(instance, 25)
    async def consume_audio():
        while True:
            await audio.recv()
    audio_task = asyncio.create_task(consume_audio())
    try:
        frames = [await video.recv()]
        started = clock.monotonic()
        await asyncio.sleep(.12)  # The RTP sender is busy encoding this frame.
        for _ in range(14):
            frames.append(await asyncio.wait_for(video.recv(), .5))
            assert video._pending.qsize() <= 1
            await asyncio.sleep(.003)
        elapsed = clock.monotonic()-started
        assert frames[1].pts >= 7200, 'A slow sender must receive the current PCM picture'
        assert len({frame.pts for frame in frames}) == len(frames)
        assert elapsed < .61, f'15 frames lost their 25 FPS cadence after one short stall: {elapsed:.4f}s'
    finally:
        audio_task.cancel()
        await asyncio.gather(audio_task, return_exceptions=True)
        audio.stop()
        await video.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_one_frame_send_buffer_absorbs_video_encoding_variation(monkeypatch):
    import av
    import fractions
    from aiortc.codecs import vpx
    from backend import clock
    from backend.transport.photo_webrtc import PhotoAudioTrack, PhotoVideoTrack
    from backend.transport.video_quality import LocalVp8Encoder
    instance = session()
    audio, video = PhotoAudioTrack(instance), PhotoVideoTrack(instance, 25)
    encoder = LocalVp8Encoder(4_000_000)
    warm = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format='rgb24')
    warm.pts, warm.time_base = 0, fractions.Fraction(1, 90000)
    await asyncio.to_thread(encoder.encode, warm)
    instance.timeline.read_audio(4800)
    original, calls = vpx.Vp8Encoder.encode, 0
    def variable_encode(self, frame, force_keyframe=False):
        nonlocal calls
        calls += 1
        if calls == 13:
            time.sleep(.065)
        return original(self, frame, force_keyframe)
    monkeypatch.setattr(vpx.Vp8Encoder, 'encode', variable_encode)
    encoder.send_at = lambda: video.send_at
    async def consume_audio():
        while True:
            await audio.recv()
    audio_task = asyncio.create_task(consume_audio())
    sent = []
    try:
        for _ in range(16):
            frame = await asyncio.wait_for(video.recv(), .5)
            await asyncio.to_thread(encoder.encode, frame)
            sent.append(clock.monotonic())
        gap = max(b-a for a, b in zip(sent, sent[1:]))
        assert gap < .09, f'Encoder timing leaked into RTP transmission: {gap:.4f}s gap'
    finally:
        audio_task.cancel()
        await asyncio.gather(audio_task, return_exceptions=True)
        audio.stop()
        await video.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_interrupt_replaces_unsent_frame_without_spending_another_period():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    track = PhotoVideoTrack(instance, 5)
    try:
        first = await track.recv()
        instance.timeline.read_audio(9600)
        async def pending_ready():
            while track._pending.empty():
                await asyncio.sleep(.005)
        await asyncio.wait_for(pending_ready(), .5)
        instance.generation += 1
        track.clear_pending()
        # The discarded frame was never sent. Its replacement can use the
        # same PCM position and must not wait for the next 200ms frame slot.
        replacement = await asyncio.wait_for(track.recv(), .08)
        assert replacement.pts == 18000 > first.pts
        instance.timeline.read_audio(9600)
        following = asyncio.create_task(track.recv())
        await asyncio.sleep(.05)
        assert not following.done(), 'Replacement must retain the original cadence'
        next_frame = await asyncio.wait_for(following, .3)
        assert next_frame.pts == 36000 > replacement.pts
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_second_interrupt_during_replacement_decode_keeps_refresh_request():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    original = instance.timeline.video_frame
    decoding, release = threading.Event(), threading.Event()
    calls = 0
    def snapshot():
        nonlocal calls
        calls += 1
        frame = original()
        if calls == 3:
            decoding.set()
            assert release.wait(.5)
        return frame
    instance.timeline.video_frame = snapshot
    track = PhotoVideoTrack(instance, 2)
    try:
        await track.recv()
        instance.timeline.read_audio(9600)
        async def pending_ready():
            while track._pending.empty():
                await asyncio.sleep(.005)
        await asyncio.wait_for(pending_ready(), 1)
        instance.generation += 1
        track.clear_pending()
        waiting = asyncio.create_task(track.recv())
        assert await asyncio.to_thread(decoding.wait, .3)
        instance.generation += 1
        track.clear_pending()
        release.set()
        frame = await asyncio.wait_for(waiting, .15)
        assert frame.pts == 18000 and calls >= 4
    finally:
        release.set()
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_late_replacement_does_not_consume_the_next_scheduled_slot():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    track = PhotoVideoTrack(instance, 5)
    try:
        await track.recv()
        instance.timeline.read_audio(9600)
        async def pending_ready():
            while track._pending.empty():
                await asyncio.sleep(.005)
        await asyncio.wait_for(pending_ready(), .5)
        instance.generation += 1
        # Reproduce a generation-file write spanning the next 200ms deadline.
        time.sleep(.21)
        instance.timeline.read_audio(9600)
        track.clear_pending()
        replacement = await asyncio.wait_for(track.recv(), .1)
        instance.timeline.read_audio(1920)
        following = await asyncio.wait_for(track.recv(), .08)
        assert following.pts > replacement.pts
        assert track._pending.qsize() <= 1
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_video_timestamps_follow_consumed_pcm_instead_of_decode_wall_time():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    instance.timeline.read_audio(48000)
    track = PhotoVideoTrack(instance, 25)
    try:
        first = await track.recv()
        assert first.pts == 90000, 'Video and audio must use the same one-second PCM position'
        instance.timeline.read_audio(4800)
        second = await track.recv()
        assert second.pts == 99000
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_decode_stall_never_publishes_duplicate_pcm_timestamps():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    positions = iter([0, 7680, 7680, 8640, 10560])
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(.16)
        pts = next(positions, 12480 + calls * 960)
        return np.zeros((16, 16, 3), np.uint8), pts

    instance.timeline.video_frame = snapshot
    track = PhotoVideoTrack(instance, 25)
    try:
        frames = [await asyncio.wait_for(track.recv(), .5) for _ in range(4)]
        timestamps = [frame.pts for frame in frames]
        assert timestamps == sorted(set(timestamps)), timestamps
        assert timestamps[:3] == [0, 14400, 16200]
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_repeated_pcm_position_does_not_shift_future_frame_deadlines(monkeypatch):
    from backend.transport import photo_webrtc
    instance = session()
    simulated = SimpleNamespace(now=0.)
    snapshots = []
    positions = iter([0, 0, 9600, 19200, 28800, 38400])
    real_sleep = asyncio.sleep
    async def sleep(seconds):
        simulated.now += seconds
        await real_sleep(0)
    async def wait_for(awaitable, timeout):
        awaitable.close()
        await sleep(timeout)
        raise asyncio.TimeoutError
    local_asyncio = vars(asyncio).copy()
    local_asyncio.update(sleep=sleep, wait_for=wait_for)
    monkeypatch.setattr(photo_webrtc, 'asyncio', SimpleNamespace(**local_asyncio))
    monkeypatch.setattr(photo_webrtc, 'clock', SimpleNamespace(monotonic=lambda: simulated.now))
    def snapshot():
        pts = next(positions, 48000)
        snapshots.append((pts, simulated.now))
        return np.zeros((16, 16, 3), np.uint8), pts
    instance.timeline.video_frame = snapshot
    track = photo_webrtc.PhotoVideoTrack(instance, 5)
    try:
        frames = [await asyncio.wait_for(track.recv(), 1) for _ in range(3)]
        assert [frame.pts for frame in frames] == [0, 18000, 36000]
        times = dict(snapshots)
        assert times[19200] == pytest.approx(.4), 'PCM retry must not move the 400ms slot'
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_video_close_wakes_a_receiver_waiting_for_next_frame():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    track = PhotoVideoTrack(instance, 25)
    await track.recv()
    waiting = asyncio.create_task(track.recv())
    await asyncio.sleep(0)
    await track.aclose()
    with pytest.raises(MediaStreamError):
        await waiting
    instance.timeline.close()


@pytest.mark.asyncio
async def test_video_close_cancels_producer_waiting_for_sender_to_take_pending_frame():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    track = PhotoVideoTrack(instance, 25)
    try:
        await track.recv()
        instance.timeline.read_audio(4800)
        async def pending_ready():
            while track._pending.empty():
                await asyncio.sleep(.005)
        await asyncio.wait_for(pending_ready(), .3)
        await asyncio.sleep(.05)
        assert not track._space.is_set()
        await asyncio.wait_for(track.aclose(), .2)
        assert track._producer is None and track._reading is None
        with pytest.raises(MediaStreamError):
            await track.recv()
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_interrupt_during_decode_retries_without_spending_next_frame_period():
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance = session()
    calls = 0

    def video_frame():
        nonlocal calls
        calls += 1
        if calls == 1:
            instance.generation += 1
        return np.full((16, 16, 3), calls, np.uint8), 0

    instance.timeline.video_frame = video_frame
    track = PhotoVideoTrack(instance, 1)
    try:
        frame = await asyncio.wait_for(track.recv(), timeout=.5)
        assert frame.to_ndarray(format='rgb24')[0, 0, 0] == 2
        assert instance.video_frames_dropped == 1
    finally:
        await track.aclose()
        instance.timeline.close()


@pytest.mark.asyncio
async def test_microphone_resamples_to_mono_and_stops_at_thirty_seconds():
    import av
    from backend.transport.photo_webrtc import consume_microphone
    instance = session()
    instance.recording = True
    instance.recorded_samples = 30 * 16000 - 30
    seen = []
    instance.emit = lambda *args, **kwargs: seen.append(args)

    class Microphone:
        count = 0

        async def recv(self):
            self.count += 1
            if self.count > 3:
                raise MediaStreamError
            frame = av.AudioFrame.from_ndarray(np.ones((1, 960), np.float32), format='flt', layout='mono')
            frame.sample_rate = 48000
            return frame

    await consume_microphone(instance, Microphone())
    assert instance.recorded_samples == 30 * 16000
    assert sum(map(len, instance.recorded)) == 30
    assert not instance.recording and seen
    instance.timeline.close()

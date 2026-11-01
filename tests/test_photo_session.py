import asyncio
import json
import threading
import time
import wave
from types import SimpleNamespace

import av
import numpy as np
import pytest


def movie(path, value=40):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=25)
        stream.width = stream.height = 16
        stream.pix_fmt = 'bgr0'
        for _ in range(25):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), value, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def session(root):
    from backend.transport.photo_session import PhotoSession
    movie(root / 'data/avatars/person/portrait/cache/idle.mkv', 70)
    instance = PhotoSession(root, {'id': 'person', 'portrait': {
        'folder': 'portrait/cache', 'width': 16, 'height': 16, 'fps': 25},
        'voice': {'confirmed': True}}, {})
    events = []
    instance.channel = SimpleNamespace(readyState='open', send=lambda raw: events.append(json.loads(raw)))
    return instance, events


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), 2)


def clients(instance, text='你好！', emotion='happy', render=None):
    from backend.emotion import Reply
    calls = []
    reply_text = text

    async def answer(text, history):
        return Reply(reply_text, emotion)

    async def voice(text, folder, profile, **kwargs):
        calls.append(('voice', kwargs))
        return np.full(2400, .2, np.float32), 24000

    async def portrait(avatar_id, session_id, generation, index, emotion):
        from backend.model_jobs import sentence_paths
        calls.append(('portrait', {'emotion': emotion, 'generation': generation}))
        audio, video = sentence_paths(instance.root, session_id, generation, index)
        with wave.open(str(audio)) as wav:
            assert wav.getnframes() == 2400 and wav.getframerate() == 24000
        if render:
            await render(generation, video)
        else:
            movie(video, 120)
        return {'report': {}}

    instance.dialogue = SimpleNamespace(answer=answer)
    instance.voice = SimpleNamespace(synthesize=voice)
    instance.portrait = SimpleNamespace(animate=portrait)
    return calls


def test_constructor_works_without_running_event_loop(tmp_path):
    instance, _ = session(tmp_path)
    assert instance.timeline.image()[0, 0, 0] == 70
    assert instance.peer is None and not instance.closed
    assert json.loads((instance.folder / 'generation.json').read_text())['closed'] is False
    asyncio.run(instance.close())


@pytest.mark.asyncio
async def test_sentence_publishes_only_after_video_and_uses_one_emotion(tmp_path):
    instance, events = session(tmp_path)
    release = asyncio.Event()

    async def render(generation, video):
        await release.wait()
        movie(video, 120)

    calls = clients(instance, render=render)
    task = asyncio.create_task(instance.respond('你好'))
    await until(lambda: len(calls) == 2)
    assert instance.timeline.read_audio(960)[0].max() == 0
    release.set()
    await until(lambda: instance.timeline.busy)
    assert [call[1]['emotion'] for call in calls] == ['happy', 'happy']
    assert instance.timeline.read_audio(4800)[0].mean() == pytest.approx(.2, abs=.001)
    instance.timeline.read_audio(960)
    await asyncio.wait_for(task, 2)
    assert any(event['type'] == 'turn_complete' for event in events)
    assert not list(instance.folder.glob('*.wav')) and not list(instance.folder.glob('*.mkv'))
    await instance.close()


@pytest.mark.asyncio
async def test_opening_sentence_video_does_not_stall_media_loop(tmp_path,monkeypatch):
    from backend.transport import photo_session
    instance,_=session(tmp_path);clients(instance)
    original=photo_session.SentenceClip
    def slow_open(*args,**kwargs):
        time.sleep(.2)
        return original(*args,**kwargs)
    monkeypatch.setattr(photo_session,'SentenceClip',slow_open)
    stamps=[]
    async def heartbeat():
        while True:
            stamps.append(time.monotonic());await asyncio.sleep(.005)
    pulse=asyncio.create_task(heartbeat())
    task=asyncio.create_task(instance.respond('你好'))
    try:
        await until(lambda:instance.timeline.busy)
        instance.timeline.read_audio(4800);instance.timeline.read_audio(960)
        await asyncio.wait_for(task,2)
        assert max(b-a for a,b in zip(stamps,stamps[1:]))<.08
    finally:
        pulse.cancel();await asyncio.gather(pulse,return_exceptions=True)
        await instance.close()


@pytest.mark.asyncio
async def test_interrupt_during_sentence_open_drains_and_closes_late_reader(tmp_path,monkeypatch):
    from backend.transport import photo_session
    instance,_=session(tmp_path);clients(instance)
    entered,release=threading.Event(),threading.Event()
    original=photo_session.SentenceClip;opened=[]
    def delayed_open(*args,**kwargs):
        clip=original(*args,**kwargs);opened.append(clip)
        entered.set()
        if not release.wait(2):raise TimeoutError('test did not release the sentence reader')
        return clip
    monkeypatch.setattr(photo_session,'SentenceClip',delayed_open)
    instance.message(json.dumps({'type':'chat_text','text':'旧问题'}))
    closing=None
    try:
        await until(entered.is_set)
        instance.interrupt()
        assert not instance.timeline.busy
        closing=asyncio.create_task(instance.close())
        await asyncio.sleep(.03)
        assert not closing.done(),'Close must drain a sentence reader still opening in a worker'
        release.set();await asyncio.wait_for(closing,2)
        assert len(opened)==1 and opened[0].closed
        assert not list(instance.folder.glob('*.mkv')) and not list(instance.folder.glob('*.wav'))
        assert not instance._model_tasks
    finally:
        release.set()
        await instance.close()
        if closing:await closing


@pytest.mark.asyncio
async def test_interrupt_during_model_job_discards_late_output_and_new_turn_plays(tmp_path):
    instance, events = session(tmp_path)
    release = asyncio.Event()
    old_generation = []

    async def render(generation, video):
        if not old_generation:
            old_generation.append(generation)
            await release.wait()
        movie(video, 120)

    calls = clients(instance, render=render)
    instance.message(json.dumps({'type': 'chat_text', 'text': '旧问题'}))
    await until(lambda: len(calls) == 2)
    old = old_generation[0]
    instance.message(json.dumps({'type': 'interrupt'}))
    marker = json.loads((instance.folder / 'generation.json').read_text())
    assert marker['generation'] != old and not instance.timeline.busy
    instance.message(json.dumps({'type': 'chat_text', 'text': '新问题'}))
    await until(lambda: instance.timeline.busy)
    assert instance.timeline.read_audio(4800)[0].mean() == pytest.approx(.2, abs=.001)
    instance.timeline.read_audio(960)
    release.set()
    await until(lambda: not list(instance.folder.glob(f'{old}_*')))
    assert not instance.timeline.busy
    assert len([event for event in events if event['type'] == 'turn_complete']) <= 1
    await instance.close()


@pytest.mark.asyncio
async def test_pending_sentences_are_bounded_before_further_inference(tmp_path):
    instance, _ = session(tmp_path)
    calls = clients(instance, text='第一句。第二句。第三句。第四句。')
    instance.message(json.dumps({'type': 'chat_text', 'text': '说四句'}))
    await until(lambda: instance.timeline.queue_depth == 2)
    await asyncio.sleep(.08)
    assert len([call for call in calls if call[0] == 'voice']) == 2
    instance.timeline.read_audio(4800)
    await until(lambda: len([call for call in calls if call[0] == 'voice']) == 3)
    await instance.close()


@pytest.mark.asyncio
async def test_recording_transcribes_and_closed_session_cannot_start_work(tmp_path):
    instance, events = session(tmp_path)
    calls = clients(instance)
    instance.dialogue.transcribe = lambda audio: '麦克风文字' if len(audio) == 4000 else ''
    instance.message(json.dumps({'type': 'start_recording'}))
    instance.recorded = [np.full(4000, .1, np.float32)]
    instance.recorded_samples = 4000
    instance.message(json.dumps({'type': 'stop_recording'}))
    await until(lambda: instance.timeline.busy)
    assert any(event.get('text') == '麦克风文字' for event in events)
    await instance.close()
    instance.message(json.dumps({'type': 'chat_text', 'text': '关闭后'}))
    await asyncio.sleep(.02)
    assert len(calls) == 2 and not instance.timeline.busy
    assert json.loads((instance.folder / 'generation.json').read_text())['closed'] is True


@pytest.mark.asyncio
async def test_close_drains_late_model_and_keeps_unowned_files(tmp_path):
    instance, _ = session(tmp_path)
    release = asyncio.Event()
    unrelated = instance.folder / 'keep.txt'
    unrelated.write_text('preserve')

    async def render(generation, video):
        await release.wait()
        movie(video)

    calls = clients(instance, render=render)
    instance.message(json.dumps({'type': 'chat_text', 'text': '问题'}))
    await until(lambda: len(calls) == 2)
    closing = asyncio.create_task(instance.close())
    await asyncio.sleep(.02)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 2)
    assert unrelated.read_text() == 'preserve'
    assert not list(instance.folder.glob('*.mkv')) and not list(instance.folder.glob('*.wav'))
    await instance.close()


@pytest.mark.asyncio
async def test_interrupt_defers_cleanup_while_worker_has_audio_open(tmp_path):
    instance, _ = session(tmp_path)
    release = asyncio.Event()
    opened = asyncio.Event()

    async def render(generation, video):
        with wave.open(str(video.with_suffix('.wav'))) as audio:
            opened.set()
            await release.wait()
            assert audio.getnframes() == 2400
        movie(video)

    clients(instance, render=render)
    instance.message(json.dumps({'type': 'chat_text', 'text': '问题'}))
    await asyncio.wait_for(opened.wait(), 2)
    instance.message(json.dumps({'type': 'interrupt'}))
    release.set()
    await instance.close()
    assert not list(instance.folder.glob('*.wav'))
    assert not list(instance.folder.glob('*.mkv'))


@pytest.mark.asyncio
async def test_finished_turn_files_are_not_rechecked_on_every_future_interrupt(tmp_path, monkeypatch):
    from pathlib import Path
    from backend.model_jobs import sentence_paths
    instance, _ = session(tmp_path)
    original = Path.unlink
    attempts = []
    def unlink(path, *args, **kwargs):
        if path.parent == instance.folder and path.suffix in {'.wav', '.mkv'}:
            attempts.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    try:
        for _ in range(10):
            generation = instance.generation
            paths = sentence_paths(instance.root, instance.id, generation, 0)
            instance._owned_paths[generation] = set(paths)
            for path in paths:
                path.write_bytes(b'temporary')
            instance.interrupt()
            assert all(not path.exists() for path in paths)
        assert len(attempts) == 20, 'Completed paths must not add filesystem work to later turns'
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_cleanup_retains_ownership_until_late_model_writer_finishes(tmp_path):
    from backend.model_jobs import sentence_paths
    instance, _ = session(tmp_path)
    generation = instance.generation
    paths = sentence_paths(instance.root, instance.id, generation, 0)
    instance._owned_paths[generation] = set(paths)
    entered, release = asyncio.Event(), asyncio.Event()
    async def late_writer():
        entered.set()
        await release.wait()
        for path in paths:
            path.write_bytes(b'late temporary result')
    job = asyncio.create_task(instance._model(late_writer(), generation))
    try:
        await entered.wait()
        instance.interrupt()
        release.set()
        await job
        await asyncio.sleep(0)
        assert all(not path.exists() for path in paths)
    finally:
        release.set()
        await instance.close()
        await asyncio.gather(job, return_exceptions=True)


@pytest.mark.asyncio
async def test_busy_path_keeps_cleanup_ownership_after_workers_finish(tmp_path, monkeypatch):
    from pathlib import Path
    from backend.model_jobs import sentence_paths
    instance, _ = session(tmp_path)
    generation = instance.generation
    audio, video = sentence_paths(instance.root, instance.id, generation, 0)
    instance._owned_paths[generation] = {audio, video}
    audio.write_bytes(b'locked temporary')
    original = Path.unlink
    locked = True
    def unlink(path, *args, **kwargs):
        if path == audio and locked:
            raise PermissionError('temporary file is still open')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    try:
        instance.interrupt()
        assert audio.exists()
        locked = False
        instance.interrupt()
        assert not audio.exists()
    finally:
        locked = False
        await instance.close()


@pytest.mark.asyncio
async def test_cleanup_skips_files_already_removed_by_playback(tmp_path, monkeypatch):
    from pathlib import Path
    from backend.model_jobs import sentence_paths
    instance, _ = session(tmp_path)
    generation = instance.generation
    paths = set(sentence_paths(instance.root, instance.id, generation, 0))
    instance._owned_paths[generation] = paths
    original = Path.unlink
    attempts = []
    def unlink(path, *args, **kwargs):
        if path in paths:
            attempts.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    try:
        instance.interrupt()
        assert not attempts, 'Playback already removed these files; do not issue another delete'
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_local_negotiation_closes_all_tracks_and_peer(tmp_path, monkeypatch):
    from aiortc import RTCConfiguration, RTCPeerConnection, rtcrtpsender
    from backend.transport.photo_webrtc import negotiate
    instance, _ = session(tmp_path)
    monkeypatch.setattr(rtcrtpsender, 'get_encoder', rtcrtpsender.get_encoder)
    remote = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    try:
        remote.createDataChannel('control')
        remote.addTransceiver('video', direction='recvonly')
        remote.addTransceiver('audio', direction='recvonly')
        await remote.setLocalDescription(await remote.createOffer())
        result = await negotiate(instance, {'sdp': remote.localDescription.sdp, 'type': 'offer'})
        assert result['type'] == 'answer' and 'm=audio' in result['sdp'] and 'm=video' in result['sdp']
        peer, video, audio = instance.peer, instance.video_track, instance.audio_track
        await instance.close()
        assert peer.connectionState == 'closed'
        assert video.readyState == audio.readyState == 'ended'
        assert not instance._transport_tasks
    finally:
        await instance.close()
        await remote.close()


@pytest.mark.asyncio
async def test_bad_offer_closes_session_after_tracks_are_created(tmp_path, monkeypatch):
    from aiortc import rtcrtpsender
    from backend.transport.photo_webrtc import negotiate
    instance, _ = session(tmp_path)
    monkeypatch.setattr(rtcrtpsender, 'get_encoder', rtcrtpsender.get_encoder)
    with pytest.raises(KeyError):
        await negotiate(instance, {'type': 'offer'})
    assert instance.closed and instance.peer is None
    assert instance.video_track is None and instance.audio_track is None


@pytest.mark.asyncio
async def test_cancelled_close_caller_does_not_abandon_model_or_idle_reader(tmp_path):
    instance, _ = session(tmp_path)
    release = asyncio.Event()

    async def render(generation, video):
        await release.wait()
        movie(video)

    calls = clients(instance, render=render)
    instance.message(json.dumps({'type': 'chat_text', 'text': '问题'}))
    await until(lambda: len(calls) == 2)
    closing = asyncio.create_task(instance.close())
    await until(lambda: instance.closed)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await instance.close()
    assert instance.timeline._idle_video is None
    assert not list(instance.folder.glob('*.mkv')) and not list(instance.folder.glob('*.wav'))


@pytest.mark.asyncio
async def test_interrupt_removes_speaking_frame_already_in_transport_queue(tmp_path):
    from backend.transport.photo_timeline import SentenceClip
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance, _ = session(tmp_path)
    generation = instance._invalidate()
    clip = SentenceClip(np.full(48000, .2, np.float32), 48000,
                        movie(instance.folder / 'speaking.mkv', 120), delete=True)
    instance.timeline.append(generation, clip)
    instance.timeline.read_audio(960)
    track = instance.video_track = PhotoVideoTrack(instance, 25)
    try:
        assert (await track.recv()).to_ndarray(format='rgb24')[0, 0, 0] == 120
        instance.timeline.read_audio(1920)
        await until(lambda: instance.video_queue_depth == 1)
        instance.interrupt()
        instance.timeline.read_audio(1920)
        frame = await asyncio.wait_for(track.recv(), 2)
        assert frame.to_ndarray(format='rgb24')[0, 0, 0] == 70
    finally:
        await instance.close()


@pytest.mark.asyncio
async def test_interrupt_rejects_speaking_decode_that_finishes_late(tmp_path, monkeypatch):
    from backend.transport.photo_timeline import SentenceClip
    from backend.transport.photo_webrtc import PhotoVideoTrack
    instance, _ = session(tmp_path)
    generation = instance._invalidate()
    clip = SentenceClip(np.full(48000, .2, np.float32), 48000,
                        movie(instance.folder / 'speaking.mkv', 120), delete=True)
    instance.timeline.append(generation, clip)
    instance.timeline.read_audio(960)
    entered, release = threading.Event(), threading.Event()
    original = instance.timeline.video_frame

    def delayed_image():
        image, audio_pts = original()
        image = image.copy()
        if not entered.is_set():
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test did not release decoder')
        return image, audio_pts

    monkeypatch.setattr(instance.timeline, 'video_frame', delayed_image)
    track = instance.video_track = PhotoVideoTrack(instance, 25)
    waiting = asyncio.create_task(track.recv())
    try:
        await until(entered.is_set)
        instance.interrupt()
        release.set()
        frame = await asyncio.wait_for(waiting, 2)
        assert frame.to_ndarray(format='rgb24')[0, 0, 0] == 70
    finally:
        release.set()
        await instance.close()

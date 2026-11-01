"""The real acceptance runner must reject incomplete media evidence.

Only HTTP, WebRTC transport and nvidia-smi are replaced. The runner, its
concurrent consumers, timers and report persistence run unchanged.
"""
import asyncio
import json
import time
from backend import clock
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from scripts import acceptance


class LocalMedia:
    def __init__(self, *, frozen=False, metrics_failure=False, prepare_delay=0):
        self.frozen = frozen
        self.metrics_failure = metrics_failure
        self.prepare_delay = prepare_delay
        self.media_started = None
        self.speaking = False
        self.tasks = []
        self.callbacks = {}
        self.closed = self.deleted = False
        self.localDescription = SimpleNamespace(sdp='test offer')
        self.readyState = 'open'

    def on(self, name):
        def register(callback):
            self.callbacks[name] = callback
            return callback
        return register

    def createDataChannel(self, label):
        assert label == 'control'
        return self

    def addTransceiver(self, *args, **kwargs):
        pass

    async def createOffer(self):
        return self.localDescription

    async def setLocalDescription(self, _):
        pass

    async def setRemoteDescription(self, _):
        await asyncio.sleep(self.prepare_delay)
        self.media_started = clock.monotonic()
        for kind in ('video', 'audio'):
            self.callbacks['track'](LocalTrack(self, kind))

    def send(self, raw):
        data = json.loads(raw)
        if data['type'] == 'interrupt':
            self.speaking = False
        elif data['type'] == 'chat_text':
            async def reply():
                self.speaking = True
                self.callbacks['message'](json.dumps({
                    'type': 'transcript', 'role': 'assistant', 'text': '本机测试回答。'}))
                await asyncio.sleep(.08)
                self.speaking = False
                self.callbacks['message'](json.dumps({'type': 'turn_complete'}))
            self.tasks.append(asyncio.create_task(reply()))

    async def close(self):
        await asyncio.gather(*self.tasks)
        self.closed = True


class LocalTrack:
    def __init__(self, media, kind):
        self.media, self.kind, self.count = media, kind, 0

    async def recv(self):
        await asyncio.sleep(.005)
        self.count += 1
        if self.kind == 'video':
            if self.media.frozen and self.count > 4:
                await asyncio.Future()  # A stalled receiver never raises.
            return SimpleNamespace(width=512, height=512,
                                   to_image=lambda: Image.new('RGB', (512, 512)))
        value = .2 if self.media.speaking else 0
        return SimpleNamespace(to_ndarray=lambda: np.full((1, 960), value, dtype=np.float32))


def install_local_io(monkeypatch, media):
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, path, **kwargs):
            payload = ({'session_id': 'test', 'width':512,'height':512,'fps':25} if path == '/api/sessions'
                       else {'sdp': 'answer', 'type': 'answer'})
            return httpx.Response(200, json=payload, request=httpx.Request('POST', 'http://test'+path))
        async def get(self, path, **kwargs):
            return httpx.Response(503 if media.metrics_failure else 200, json={'queue_depth': 0},
                                  request=httpx.Request('GET', 'http://test'+path))
        async def delete(self, path):
            media.deleted = True
            return httpx.Response(204, request=httpx.Request('DELETE', 'http://test'+path))
    monkeypatch.setattr(acceptance.httpx, 'AsyncClient', lambda **kwargs: Client())
    monkeypatch.setattr(acceptance, 'RTCPeerConnection', lambda **kwargs: media)
    monkeypatch.setattr(acceptance, 'RTCSessionDescription', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(acceptance.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout='71\n'))


def read_report(root):
    return json.loads(next((root/'outputs/acceptance').glob('*/report.json')).read_text())


async def test_acceptance_rejects_monitor_failure(tmp_path, monkeypatch):
    install_local_io(monkeypatch, LocalMedia(metrics_failure=True))
    with pytest.raises(RuntimeError, match='monitor|503'):
        await acceptance.run(tmp_path, 'avatar', minutes=.05, rounds=0)
    assert read_report(tmp_path)['status'] == 'failed'


async def test_acceptance_rejects_video_that_stops_without_raising(tmp_path, monkeypatch):
    install_local_io(monkeypatch, LocalMedia(frozen=True))
    with pytest.raises(RuntimeError, match='video|Video'):
        await acceptance.run(tmp_path, 'avatar', minutes=.05, rounds=0)
    assert read_report(tmp_path)['status'] == 'failed'


async def test_acceptance_duration_excludes_preparation(tmp_path, monkeypatch):
    media = LocalMedia(prepare_delay=1)
    install_local_io(monkeypatch, media)
    await acceptance.run(tmp_path, 'avatar', minutes=.05, rounds=0)
    received_seconds = clock.monotonic()-media.media_started
    assert received_seconds >= 3, 'Model loading and negotiation must not shorten the media test'
    report = read_report(tmp_path)
    assert report['media_elapsed_seconds'] >= 3
    assert report['preparation_seconds'] >= 1


async def test_acceptance_saving_frame_evidence_does_not_block_media_loop(tmp_path, monkeypatch):
    install_local_io(monkeypatch, LocalMedia())
    original_save = Image.Image.save

    def slow_disk(image, path, *args, **kwargs):
        time.sleep(.25)
        return original_save(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, 'save', slow_disk)
    stamps = []

    async def heartbeat():
        while True:
            stamps.append(clock.monotonic())
            await asyncio.sleep(.005)

    pulse = asyncio.create_task(heartbeat())
    try:
        await acceptance.run(tmp_path, 'avatar', minutes=.05, rounds=0)
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)
    largest_pause = max(b-a for a, b in zip(stamps, stamps[1:]))
    assert largest_pause < .12, 'Evidence PNG writes must not stop WebRTC packet processing'
    folder = next((tmp_path/'outputs/acceptance').iterdir())
    assert (folder/'frame_00001.png').is_file()
    assert (folder/'frame_00100.png').is_file()


async def test_acceptance_rejects_failed_frame_evidence_write(tmp_path, monkeypatch):
    install_local_io(monkeypatch, LocalMedia())

    def disk_error(*args, **kwargs):
        raise OSError('evidence disk full')

    monkeypatch.setattr(Image.Image, 'save', disk_error)
    with pytest.raises(RuntimeError, match='evidence disk full'):
        await acceptance.run(tmp_path, 'avatar', minutes=.05, rounds=0)
    assert read_report(tmp_path)['status'] == 'failed'


async def test_acceptance_report_writes_do_not_block_media_loop(tmp_path, monkeypatch):
    install_local_io(monkeypatch, LocalMedia())
    original = acceptance.atomic_json
    def slow_report(*args, **kwargs):
        time.sleep(.25)
        return original(*args, **kwargs)
    monkeypatch.setattr(acceptance, 'atomic_json', slow_report)
    stamps = []
    async def heartbeat():
        while True:
            stamps.append(clock.monotonic())
            await asyncio.sleep(.005)
    pulse = asyncio.create_task(heartbeat())
    try:
        await acceptance.run(tmp_path, 'avatar', minutes=.05, rounds=0)
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)
    assert max(b-a for a, b in zip(stamps, stamps[1:])) < .12
    assert read_report(tmp_path)['status'] == 'passed'


async def test_cancelled_monitor_drains_old_report_before_final_status(tmp_path, monkeypatch):
    media = LocalMedia()
    install_local_io(monkeypatch, media)
    original = acceptance.atomic_json
    writes = []
    cleanup_observations = []
    def delayed_progress(path, value):
        if value['status'] == 'running':
            deadline = clock.monotonic()+3
            while not media.deleted and clock.monotonic()<deadline:
                time.sleep(.01)
            cleanup_observations.append((media.closed, media.deleted))
        writes.append(value['status'])
        return original(path, value)
    monkeypatch.setattr(acceptance, 'atomic_json', delayed_progress)
    await acceptance.run(tmp_path, 'avatar', minutes=0, rounds=0)
    assert writes == ['running', 'passed']
    assert cleanup_observations == [(True, True)]
    assert read_report(tmp_path)['status'] == 'passed'

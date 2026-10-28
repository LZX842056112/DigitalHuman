"""The real acceptance runner must reject incomplete media evidence.

Only HTTP, WebRTC transport and nvidia-smi are replaced. The runner, its
concurrent consumers, timers and report persistence run unchanged.
"""
import asyncio
import json
import time
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
        self.media_started = time.monotonic()
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
            payload = ({'session_id': 'test', 'camera': {'yaw': 0}} if path == '/api/sessions'
                       else {'sdp': 'answer', 'type': 'answer'})
            return httpx.Response(200, json=payload, request=httpx.Request('POST', 'http://test'+path))
        async def get(self, path, **kwargs):
            return httpx.Response(503 if media.metrics_failure else 200, json={'queue_depth': 0},
                                  request=httpx.Request('GET', 'http://test'+path))
        async def delete(self, path):
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
    received_seconds = time.monotonic()-media.media_started
    assert received_seconds >= 3, 'Model loading and negotiation must not shorten the media test'
    report = read_report(tmp_path)
    assert report['media_elapsed_seconds'] >= 3
    assert report['preparation_seconds'] >= 1

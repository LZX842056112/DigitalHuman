"""Keep WAV bytes and worker responses stable when sharing PCM16 encoding."""
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.media import write_wav, write_wav_stream
from backend.storage import sha256


@pytest.mark.parametrize('rate', [8000, 16000, 24000, 48000])
@pytest.mark.parametrize('samples, integers', [
    ([], []),
    ([0., 0., 0.], [0, 0, 0]),
    ([-2., -1., -.5, 0., .5, 1., 2.], [-32767, -32767, -16383, 0, 16383, 32767, 32767]),
])
def test_wav_file_and_stream_keep_original_pcm16_bytes(tmp_path, rate, samples, integers):
    pcm = np.asarray(samples, dtype=np.float32)
    expected = io.BytesIO()
    with wave.open(expected, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(integers, dtype='<i2').tobytes())
    stream = io.BytesIO()
    write_wav_stream(stream, pcm, rate)
    path = tmp_path/'nested/audio.wav'
    write_wav(path, pcm, rate)
    assert not stream.closed
    assert path.read_bytes() == stream.getvalue() == expected.getvalue()


@pytest.fixture
def voice_worker(tmp_path, monkeypatch):
    from backend.voice import service
    folder = tmp_path/'person'
    folder.mkdir()
    reference = folder/'reference.wav'
    reference.write_bytes(b'fixture reference')
    manifest = {'schema_version': 2, 'mode': 'photo', 'voice': {
        'confirmed': True, 'reference': reference.name, 'reference_sha256': sha256(reference)}}
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'store', SimpleNamespace(
        manifest=lambda _: manifest, avatar_dir=lambda _: folder))
    monkeypatch.setattr(service, 'expired', lambda *args: False)
    return service


def test_voice_worker_keeps_wav_body_and_metrics_headers(voice_worker, monkeypatch):
    pcm = np.asarray([-.5, 0., .5], dtype=np.float32)
    calls = []

    def synthesize(text, reference, emotion, cache, cancelled):
        calls.append((text, reference.name, emotion, cache.name, cancelled()))
        return pcm, 24000, {'generation_seconds': 1.25, 'peak_vram_mib': 512.}

    monkeypatch.setattr(voice_worker, 'engine', SimpleNamespace(synthesize=synthesize))
    with TestClient(voice_worker.app) as client:
        response = client.post('/tts', json={'avatar_id': 'person', 'text': '你好', 'emotion': 'happy'})
    expected = io.BytesIO()
    write_wav_stream(expected, pcm, 24000)
    assert response.status_code == 200
    assert response.content == expected.getvalue()
    assert response.headers['content-type'] == 'audio/wav'
    assert response.headers['x-generation-seconds'] == '1.25'
    assert response.headers['x-peak-vram-mib'] == '512.0'
    assert calls == [('你好', 'reference.wav', 'happy', 'voice', False)]


@pytest.mark.parametrize('error, status', [(ValueError, 400), (InterruptedError, 409)])
def test_voice_worker_keeps_error_status(voice_worker, monkeypatch, error, status):
    def synthesize(*args):
        raise error('fixture failure')

    monkeypatch.setattr(voice_worker, 'engine', SimpleNamespace(synthesize=synthesize))
    with TestClient(voice_worker.app) as client:
        response = client.post('/tts', json={'avatar_id': 'person', 'text': '你好'})
    assert response.status_code == status
    assert response.json() == {'detail': 'fixture failure'}

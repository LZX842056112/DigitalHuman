import wave
from pathlib import Path
import numpy as np
import pytest


def test_reference_selection_avoids_silence_and_clipping():
    from backend.media import choose_reference
    sr = 16000
    t = np.arange(sr * 6) / sr
    speech = (.15 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    pcm = np.concatenate([np.zeros(sr * 7), speech, np.ones(sr * 7)])
    start, duration = choose_reference(pcm, sr, 6)
    assert 6.5 <= start <= 7.5
    assert duration == 6


def test_reference_selection_rejects_silence():
    from backend.media import choose_reference
    from backend.errors import PipelineError
    with pytest.raises(PipelineError, match="语音"):
        choose_reference(np.zeros(16000 * 12), 16000)


def test_audio_decode_resamples_without_mutating_source(tmp_path):
    from backend.media import read_audio, write_wav
    src = tmp_path / "voice.wav"
    with wave.open(str(src), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.full(8000, 1000, dtype="<i2").tobytes())
    original = src.read_bytes()
    pcm, sr = read_audio(src, 16000)
    assert sr == 16000
    assert abs(len(pcm) - 16000) <= 2
    assert src.read_bytes() == original
    write_wav(tmp_path / "copy.wav", pcm, sr)
    with wave.open(str(tmp_path / "copy.wav")) as f:
        assert f.getframerate() == 16000


def test_audio_decode_accepts_exact_budget_and_rejects_one_extra_sample(tmp_path):
    from backend.media import read_audio,write_wav
    from backend.errors import PipelineError
    path=tmp_path/'audio.wav'
    write_wav(path,np.ones(1000,np.float32)*.1,16000)
    assert len(read_audio(path,max_samples=1000)[0])==1000
    assert len(read_audio(path,max_samples=None)[0])==1000
    write_wav(path,np.ones(1001,np.float32)*.1,16000)
    with pytest.raises(PipelineError,match='解码.*上限'):
        read_audio(path,max_samples=1000)


@pytest.mark.parametrize('flush',[False,True])
def test_audio_budget_stops_before_copying_excess_and_counts_resampler_tail(monkeypatch,flush):
    from types import SimpleNamespace
    from backend import media
    from backend.errors import PipelineError
    decoded=[];copied=[];closed=[]
    class Frame:
        def __init__(self,count):self.samples=count
        def to_ndarray(self):copied.append(self.samples);return np.zeros((1,self.samples),np.float32)
    class Container:
        streams=SimpleNamespace(audio=[object()])
        duration=None
        def __enter__(self):return self
        def __exit__(self,*args):closed.append(True)
        def decode(self,**kwargs):
            assert kwargs=={'audio':0}
            for index in range(2 if flush else 100):decoded.append(index);yield index
    monkeypatch.setattr(media.av,'open',lambda *_:Container())
    monkeypatch.setattr(media.av,'AudioResampler',lambda **k:SimpleNamespace(
        resample=lambda frame:[Frame(1 if frame is None else 4)]))
    with pytest.raises(PipelineError,match='解码.*上限'):
        media.read_audio('fixture.mp4',max_samples=8)
    assert len(decoded)==(2 if flush else 3)
    assert copied==[4,4] and closed==[True]


def reference_person(tmp_path):
    from backend.storage import Store
    from backend.media import write_wav
    photo=tmp_path/'photo.png';photo.write_bytes(b'photo')
    video=tmp_path/'voice.mp4'
    pcm=(np.sin(np.arange(8*16000)*.1)*.1).astype(np.float32)
    write_wav(video,pcm,16000)
    store=Store(tmp_path/'project');person=store.import_photo('test',photo,video)
    target=store.avatar_dir(person['id'])/'voice/reference.wav'
    target.parent.mkdir();target.write_bytes(b'previous reference')
    return store,person,target


def test_reference_write_failure_preserves_previous_recording_and_manifest(tmp_path,monkeypatch):
    from backend import media
    store,person,target=reference_person(tmp_path)
    original=target.read_bytes()
    def broken(path,*args):Path(path).write_bytes(b'partial');raise OSError('write failed')
    monkeypatch.setattr(media,'write_wav',broken)
    with pytest.raises(OSError,match='write failed'):media.prepare_voice(store,person['id'])
    assert target.read_bytes()==original
    assert list(target.parent.iterdir())==[target]
    assert store.manifest(person['id'])==person


def test_reference_decode_budget_preserves_existing_recording(tmp_path,monkeypatch):
    from backend import media
    from backend.errors import PipelineError
    store,person,target=reference_person(tmp_path)
    monkeypatch.setattr(media,'REFERENCE_AUDIO_MAX_SECONDS',1,raising=False)
    with pytest.raises(PipelineError,match='解码.*上限'):media.prepare_voice(store,person['id'])
    assert target.read_bytes()==b'previous reference'
    assert store.manifest(person['id'])==person


def test_verified_reference_cache_does_not_decode_again(tmp_path,monkeypatch):
    from backend import media
    from backend.storage import sha256
    store,person,target=reference_person(tmp_path)
    person['voice'].update(source_sha256=person['sources'][person['voice_video']],
                           reference_sha256=sha256(target),confirmed=True)
    store.write_manifest(person['id'],person)
    monkeypatch.setattr(media,'read_audio',lambda *a,**k:pytest.fail('decoded valid cached reference'))
    assert media.prepare_voice(store,person['id'])==person

import wave
from pathlib import Path

import numpy as np
import pytest


def _person(tmp_path, seconds=8, amplitude=.1):
    from backend.storage import Store
    from backend.media import write_wav
    photo=tmp_path/'photo.png';photo.write_bytes(b'photo')
    video=tmp_path/'voice.mp4'
    rate=16000
    t=np.arange(int(rate*seconds))/rate
    write_wav(video,(amplitude*np.sin(2*np.pi*180*t)).astype(np.float32),rate)
    store=Store(tmp_path/'project')
    return store,store.import_photo('test',photo,video)


def test_quiet_recording_still_yields_a_reference(tmp_path):
    from backend.media import choose_reference
    rate=16000
    t=np.arange(rate*8)/rate
    pcm=np.concatenate([np.zeros(rate),
                        (.006*np.sin(2*np.pi*200*t)).astype(np.float32)]).reshape(-1)
    start,duration=choose_reference(pcm,rate,6)
    assert .5<=start<=1.5 and duration==6


def test_most_typical_rejects_a_window_far_from_the_speaker_mean():
    from backend.media import _most_typical
    starts=[0.,2.,4.]
    vectors=[np.array([1.,0.],np.float32),np.array([.8,.6],np.float32),np.array([-1.,0.],np.float32)]
    assert _most_typical(starts,vectors)==2.


def test_reference_cleaning_normalizes_level_and_softens_edges():
    from backend.media import clean_reference
    rate=24000
    t=np.arange(rate*2)/rate
    pcm=(.02*np.sin(2*np.pi*220*t)).astype(np.float32)
    cleaned=clean_reference(pcm,rate)
    voiced=cleaned[np.abs(cleaned)>.05*float(np.max(np.abs(cleaned)))]
    assert np.sqrt(np.mean(voiced.astype(np.float64)**2))==pytest.approx(.08,abs=.005)
    assert float(np.max(np.abs(cleaned)))<=.95
    assert abs(float(cleaned[0]))<1e-3 and abs(float(cleaned[-1]))<1e-3
    assert float(np.mean(cleaned))==pytest.approx(0.,abs=1e-4)


def test_reference_choice_samples_the_whole_track(tmp_path):
    from backend.media import choose_reference
    rate=16000
    t=np.arange(rate*20)/rate
    loud=(.1*np.sin(2*np.pi*180*t[:rate*3])).astype(np.float32)
    quiet=(.05*np.sin(2*np.pi*180*t[:rate*17])).astype(np.float32)
    pcm=np.concatenate([loud,quiet])
    class Embedder:
        def __init__(self):self.windows=0
        def embedding(self,pcm,rate):
            self.windows+=1
            return np.array([1.,0.] if float(np.max(np.abs(pcm)))>.075 else [0.,1.],np.float32)
    embedder=Embedder()
    start,duration=choose_reference(pcm,rate,6,embedder=embedder)
    assert embedder.windows>=4
    assert start>=3.,'picked only from the opening seconds'


def test_prepare_voice_keeps_full_band_and_transcript(tmp_path,monkeypatch):
    from backend import asr,media
    store,person=_person(tmp_path)
    monkeypatch.setattr(asr,'transcribe',lambda root,path:(u'你好，我们可以开始聊天了。','zh'))
    manifest=media.prepare_voice(store,person['id'])
    voice=manifest['voice']
    assert voice['text']==u'你好，我们可以开始聊天了。'
    assert voice['text_language']=='zh' and voice['confirmed'] is False
    assert voice['sample_rate']==24000 and voice['duration_seconds']==6
    target=store.avatar_dir(person['id'])/'voice/reference.wav'
    with wave.open(str(target)) as handle:
        assert handle.getframerate()==24000
        pcm=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2')/32768
    assert len(pcm)/24000==pytest.approx(6,abs=.02)
    assert float(np.max(np.abs(pcm)))<=.95


def test_missing_transcript_does_not_block_preparation(tmp_path,monkeypatch):
    from backend import asr,media
    store,person=_person(tmp_path)
    def broken(root,path):raise RuntimeError('识别模型不可用')
    monkeypatch.setattr(asr,'transcribe',broken)
    manifest=media.prepare_voice(store,person['id'])
    assert manifest['voice']['text'] is None
    assert '识别模型不可用' in manifest['voice']['transcription_error']
    assert (store.avatar_dir(person['id'])/'voice/reference.wav').is_file()


def test_legacy_reference_is_rebuilt_once(tmp_path,monkeypatch):
    from backend import asr,media
    store,person=_person(tmp_path)
    monkeypatch.setattr(asr,'transcribe',lambda root,path:('第一版。','zh'))
    media.prepare_voice(store,person['id'])
    manifest=store.manifest(person['id'])
    manifest['voice'].pop('reference_version')
    store.write_manifest(person['id'],manifest)
    calls=[];original=media.read_audio
    monkeypatch.setattr(media,'read_audio',lambda *a,**k:(calls.append(1),original(*a,**k))[1])
    rebuilt=media.prepare_voice(store,person['id'])
    assert calls,'legacy reference was reused without rebuilding'
    assert rebuilt['voice']['reference_version']==media.REFERENCE_VERSION


def test_unchanged_source_skips_decode_and_transcription(tmp_path,monkeypatch):
    from backend import asr,media
    from backend.storage import sha256
    store,person=_person(tmp_path)
    monkeypatch.setattr(asr,'transcribe',lambda root,path:('第一版。','zh'))
    media.prepare_voice(store,person['id'])
    manifest=store.manifest(person['id'])
    monkeypatch.setattr(asr,'transcribe',lambda *a: pytest.fail('re-ran recognition'))
    monkeypatch.setattr(media,'read_audio',lambda *a,**k: pytest.fail('decoded again'))
    assert media.prepare_voice(store,person['id'])==manifest
    assert manifest['voice']['reference_sha256']==sha256(
        store.avatar_dir(person['id'])/'voice/reference.wav')

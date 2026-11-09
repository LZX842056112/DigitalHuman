import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('mode,emotion,text,expected',[
    ('hybrid','neutral','你好。',('zero_shot','你好。')),
    ('hybrid','happy','你好。',('instruct','')),
    ('hybrid','neutral','',('instruct','')),
    ('zero_shot','sad','你好。',('zero_shot','你好。')),
    ('instruct','neutral','你好。',('instruct','')),
])
def test_clone_mode_mapping(monkeypatch,mode,emotion,text,expected):
    from backend.voice import service
    monkeypatch.setattr(service,'CLONE_MODE',mode)
    assert service.clone_settings(emotion,text)==expected


def test_cache_key_separates_transcript_and_mode():
    from backend.voice.engine import voice_cache_key,zero_shot_prompt_text
    base=voice_cache_key('voice','neutral','model')
    assert base!=voice_cache_key('voice','neutral','model','你好。')
    assert voice_cache_key('voice','neutral','model','你好。')!=voice_cache_key(
        'voice','neutral','model','你好。','zero_shot')
    assert zero_shot_prompt_text('  你好。 ')=='You are a helpful assistant.<|endofprompt|>你好。'


def _engine(tmp_path):
    from backend.voice.engine import VoiceEngine
    calls=[];model=SimpleNamespace(lock=threading.Lock())
    def instruct(*args,**kwargs):
        calls.append(('instruct',args,kwargs));yield {'tts_speech':None}
    def zero(*args,**kwargs):
        calls.append(('zero',args,kwargs));yield {'tts_speech':None}
    engine=VoiceEngine(tmp_path)
    engine.voice=SimpleNamespace(model=model,inference_instruct2=instruct,
                                 inference_zero_shot=zero)
    return engine,calls


def test_zero_shot_path_skips_the_emotion_instruction(tmp_path):
    from backend.emotion import voice_instruction
    engine,calls=_engine(tmp_path)
    reference=tmp_path/'reference.wav'
    list(engine._native_chunks('你好。','sad',reference,'speaker',stream=True,mode='zero_shot'))
    assert calls[0][0]=='zero'
    assert calls[0][1]==('你好。','',str(reference))
    assert calls[0][2]['zero_shot_spk_id']=='speaker'
    list(engine._native_chunks('你好。','sad',reference,'speaker',stream=True))
    assert calls[1][0]=='instruct'
    assert calls[1][1][1]==voice_instruction('sad')


def test_speaker_seed_uses_transcript_for_zero_shot(tmp_path):
    import torch
    from backend.storage import sha256
    from backend.voice.engine import VoiceEngine,voice_cache_key,zero_shot_prompt_text
    class Frontend:
        def __init__(self):self.spk2info={};self.seeds=[]
        def add_zero_shot_spk(self,seed,reference,key):
            self.seeds.append(seed);self.spk2info[key]={'embedding':torch.ones(2)}
    engine=VoiceEngine(tmp_path);frontend=Frontend()
    engine.voice=SimpleNamespace(frontend=frontend,add_zero_shot_spk=frontend.add_zero_shot_spk)
    engine.revision='model'
    reference=tmp_path/'reference.wav';reference.write_bytes(b'reference')
    key=engine._speaker(reference,'neutral',tmp_path,'你好。','zero_shot')
    assert frontend.seeds==[zero_shot_prompt_text('你好。')]
    assert key==voice_cache_key(sha256(reference),'neutral','model','你好。','zero_shot')
    assert engine._speaker(reference,'neutral',tmp_path)!=key
    assert frontend.seeds[1]!='' and '<|endofprompt|>' in frontend.seeds[1]


def test_health_reports_the_configured_mode():
    from fastapi.testclient import TestClient
    from backend.voice import service
    assert TestClient(service.app).get('/health').json()['clone_mode']==service.CLONE_MODE

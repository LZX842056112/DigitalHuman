"""One recognizer is published only after successful instance initialization."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys
import threading

import numpy as np
import pytest

from backend.dialogue.client import Dialogue


def dialogue(tmp_path):
    model=tmp_path/'models/asr/small/model.bin'
    model.parent.mkdir(parents=True);model.write_bytes(b'fixture')
    return Dialogue(tmp_path,{})


def recognizer():
    return SimpleNamespace(transcribe=lambda *a,**k:([SimpleNamespace(text='测试')],None))


def test_concurrent_first_transcriptions_construct_one_recognizer(tmp_path,monkeypatch):
    entered,second_entered,release=threading.Event(),threading.Event(),threading.Event()
    constructions=[]
    def construct(*args,**kwargs):
        constructions.append(kwargs)
        if len(constructions)>1:second_entered.set()
        entered.set();assert release.wait(3)
        return recognizer()
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=construct))
    instance=dialogue(tmp_path)
    with ThreadPoolExecutor(2) as executor:
        first=executor.submit(instance.transcribe,np.zeros(16000,np.float32))
        try:
            assert entered.wait(2)
            second=executor.submit(instance.transcribe,np.zeros(16000,np.float32))
            second_entered.wait(.15)
        finally:release.set()
        assert first.result(timeout=2)==second.result(timeout=2)=='测试'
    assert len(constructions)==1,'Concurrent calls loaded duplicate recognizers'
    assert constructions[0]=={'device':'cpu','compute_type':'int8','cpu_threads':4,'local_files_only':True}


def test_failed_recognizer_initialization_allows_retry(tmp_path,monkeypatch):
    calls=[]
    def construct(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:raise RuntimeError('load failed')
        return recognizer()
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=construct))
    instance=dialogue(tmp_path)
    with pytest.raises(RuntimeError,match='load failed'):instance.transcribe(np.zeros(10))
    assert instance._recognizer is None
    assert instance.transcribe(np.zeros(10))=='测试'
    assert len(calls)==2

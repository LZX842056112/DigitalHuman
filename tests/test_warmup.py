import threading

from backend import warmup


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv('DIGITALHUMAN_WARMUP',raising=False)
    called=[]
    assert warmup.start(lambda:called.append(1),threading.Lock(),'test-warmup') is None
    assert called==[]


def test_runs_once_under_service_lock(monkeypatch):
    monkeypatch.setenv('DIGITALHUMAN_WARMUP','1')
    lock=threading.Lock();called=[]
    thread=warmup.start(lambda:called.append(1),lock,'test-warmup')
    thread.join(10)
    assert called==[1]
    assert not lock.locked()


def test_failure_keeps_service_usable(monkeypatch):
    monkeypatch.setenv('DIGITALHUMAN_WARMUP','1')
    lock=threading.Lock()
    def boom():raise RuntimeError('no gpu')
    thread=warmup.start(boom,lock,'test-warmup')
    thread.join(10)
    assert not lock.locked()

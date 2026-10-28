import threading
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from backend.avatar.lhmpp import LHMRenderer


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_explicit_unload_releases_device_assets_even_with_external_loader_references(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    before=torch.cuda.memory_allocated() if device=='cuda' else 0
    renderer=LHMRenderer.__new__(LHMRenderer)
    renderer.lock=threading.RLock();renderer._closed=False
    renderer.model=SimpleNamespace(renderer=torch.nn.Linear(1024,1024).to(device),
                                   neural_renderer=torch.nn.Linear(1024,1024).to(device))
    class Cache:
        def __init__(self): self.values=[torch.ones(256,256,device=device)]
        def close(self): self.values.clear()
    renderer.runtime_cache=Cache()
    renderer.state={'features':torch.ones(256,256,device=device)}
    # A retained loader frame can own these aliases after a session ends.
    held_renderer=renderer;held_model=renderer.model;held_state=renderer.state;held_cache=renderer.runtime_cache
    renderer.close();renderer.close()
    assert held_state=={} and not held_cache.values
    assert held_renderer.model is None and held_renderer.runtime_cache is None
    assert next(held_model.renderer.parameters()).device.type=='cpu'
    assert next(held_model.neural_renderer.parameters()).device.type=='cpu'
    with pytest.raises(RuntimeError,match='卸载'):
        held_renderer.render('idle',0,{},np.zeros(100),np.zeros(3))
    if device=='cuda':
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated()<=before+1024*1024


def test_unload_also_closes_the_independent_head_decoder():
    renderer=LHMRenderer.__new__(LHMRenderer)
    renderer.lock=threading.RLock();renderer._closed=False;renderer.runtime_cache=None
    renderer.model=SimpleNamespace(renderer=torch.nn.Linear(1,1),neural_renderer=torch.nn.Linear(1,1))
    renderer.state={};closed=[]
    renderer.head_refiner=SimpleNamespace(close=lambda:closed.append(True))
    renderer.close();renderer.close()
    assert closed==[True] and renderer.head_refiner is None


def test_failed_session_creation_unloads_successfully_loaded_avatar(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from backend.storage import Store
    from backend.api.app import create_app
    from backend.transport import session
    from backend.avatar.lease import GpuLease
    unloaded=[]
    class Renderer:
        def close(self): unloaded.append(True)
    class BrokenSession:
        def __init__(self,*args): raise RuntimeError('Session initialization failed')
    monkeypatch.setattr(Store,'require_ready',lambda *_:{'id':'avatar'})
    monkeypatch.setattr(LHMRenderer,'load',lambda *_:Renderer())
    monkeypatch.setattr(session,'Session',BrokenSession)
    with TestClient(create_app(tmp_path),raise_server_exceptions=False) as client:
        response=client.post('/api/sessions',json={'avatar_id':'avatar'})
    assert response.status_code==500
    assert unloaded==[True],'A failed session left the loaded GPU avatar alive'
    lease=GpuLease(tmp_path).acquire();lease.release()


async def test_cancelled_load_unloads_its_result_before_releasing_gpu(tmp_path,monkeypatch):
    import asyncio
    from backend.api.app import create_app
    from backend.avatar.lease import GpuLease
    from backend.errors import NotReady
    from backend.storage import Store
    entered=threading.Event();release=threading.Event();unloaded=[]
    class Renderer:
        def close(self): unloaded.append(True)
    def load(*args):
        entered.set();release.wait(3)
        return Renderer()
    monkeypatch.setattr(Store,'require_ready',lambda *_:{'id':'avatar'})
    monkeypatch.setattr(LHMRenderer,'load',load)
    app=create_app(tmp_path)
    endpoint=next(route.endpoint for route in app.routes if getattr(route,'path',None)=='/api/sessions' and 'POST' in route.methods)
    pending=asyncio.create_task(endpoint(SimpleNamespace(avatar_id='avatar')))
    try:
        assert await asyncio.to_thread(entered.wait,1)
        pending.cancel();await asyncio.sleep(.02)
        with pytest.raises(NotReady): GpuLease(tmp_path).acquire()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError): await pending
    assert unloaded==[True],'A cancelled request left its finished model load on the GPU'
    lease=GpuLease(tmp_path).acquire();lease.release()


@pytest.mark.parametrize('stage',['state','warmup'])
def test_failed_renderer_load_releases_networks_already_moved_to_gpu(tmp_path,monkeypatch,stage):
    from backend.avatar import lhmpp,runtime_cache
    class Network:
        def __init__(self): self.on_gpu=False
        def cuda(self): self.on_gpu=True;return self
        def cpu(self): self.on_gpu=False;return self
    model=SimpleNamespace(renderer=Network(),neural_renderer=Network())
    state={'features':torch.ones(1)}
    def read_state(*args):
        if stage=='state': raise ValueError('Invalid avatar tensor')
        return state
    def render(*args): raise ValueError('Failed renderer warmup')
    monkeypatch.setattr(lhmpp,'load_model',lambda *_:model)
    monkeypatch.setattr(lhmpp,'load_state',read_state)
    monkeypatch.setattr(runtime_cache,'install',lambda *_:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(LHMRenderer,'render',render)
    with pytest.raises(ValueError): LHMRenderer.load(tmp_path,tmp_path)
    assert not model.renderer.on_gpu and not model.neural_renderer.on_gpu
    if stage=='warmup': assert not state

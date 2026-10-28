from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import pytest


def test_failed_state_commit_preserves_previous_snapshot(tmp_path, monkeypatch):
    import torch
    from backend.avatar import state
    state.save_state(tmp_path, {'value':torch.tensor([1.])})
    def fail(*args, **kwargs): raise OSError('simulated disk failure')
    monkeypatch.setattr(state,'atomic_json',fail)
    with pytest.raises(OSError): state.save_state(tmp_path, {'value':torch.tensor([2.])})
    assert state.load_state(tmp_path)['value'].item()==1


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_nonfinite_reconstruction_cannot_replace_a_valid_state(tmp_path,invalid):
    import torch
    from backend.avatar import state
    state.save_state(tmp_path,{'gaussian_opacity':torch.tensor([.5])})
    original=(tmp_path/'state.json').read_bytes()
    with pytest.raises(ValueError,match='无效数值'):
        state.save_state(tmp_path,{'gaussian_opacity':torch.tensor([invalid])})
    assert (tmp_path/'state.json').read_bytes()==original
    assert state.load_state(tmp_path)['gaussian_opacity'].item()==.5


def test_loading_a_checksum_valid_nan_state_is_rejected(tmp_path):
    import torch
    from safetensors.torch import save_file
    from backend.avatar import state
    from backend.storage import read_json,atomic_json,sha256
    state.save_state(tmp_path,{'gaussian_opacity':torch.tensor([.5])})
    schema=read_json(tmp_path/'state.json')
    tensor_path=tmp_path/schema['tensor_file']
    save_file({'tensor_00000':torch.tensor([float('nan')])},str(tensor_path))
    schema['tensor_sha256']=sha256(tensor_path)
    atomic_json(tmp_path/'state.json',schema)
    with pytest.raises(ValueError,match='无效数值'):
        state.load_state(tmp_path)


def test_head_optimizer_deltas_are_fp32_and_finite_for_zero_gradient():
    import torch
    from backend.avatar.calibrate import head_deltas
    features=torch.ones(1,4,3,dtype=torch.float16)
    offsets=torch.zeros(4,3,dtype=torch.float16)
    feature_delta,offset_delta=head_deltas(features,offsets,torch.tensor([0,1]))
    assert feature_delta.dtype==offset_delta.dtype==torch.float32
    optimizer=torch.optim.Adam([feature_delta,offset_delta],lr=.001)
    (feature_delta.sum()*0+offset_delta.sum()*0).backward()
    optimizer.step()
    assert torch.isfinite(feature_delta).all() and torch.isfinite(offset_delta).all()


def test_session_keeps_gpu_reservation_until_close_finishes(tmp_path, monkeypatch):
    import asyncio
    from fastapi.testclient import TestClient
    from backend.storage import Store
    from backend.avatar.lhmpp import LHMRenderer
    from backend.transport import session
    from backend.api.app import create_app
    entered,released=threading.Event(),threading.Event()
    class Renderer:
        default_camera={}
    class FakeSession:
        def __init__(self,*args): self.id='test_session';self.renderer=args[2];self.closed=False
        async def close(self):
            entered.set()
            await asyncio.to_thread(released.wait,5)
            self.closed=True
    monkeypatch.setattr(Store,'require_ready',lambda *_:{'id':'avatar'})
    monkeypatch.setattr(LHMRenderer,'load',lambda *_:Renderer())
    monkeypatch.setattr(session,'Session',FakeSession)
    with TestClient(create_app(tmp_path)) as client, ThreadPoolExecutor(1) as executor:
        assert client.post('/api/sessions',json={'avatar_id':'avatar'}).status_code==201
        pending=executor.submit(client.delete,'/api/sessions/test_session')
        assert entered.wait(3)
        try:
            assert client.post('/api/sessions',json={'avatar_id':'avatar'}).status_code==409
        finally:
            released.set()
            pending.result(5)


def test_calibration_autocast_accepts_real_half_cache_and_backpropagates():
    import torch
    from types import SimpleNamespace
    from backend.avatar.lhmpp import LHMRenderer
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    class Raster:
        def get_single_view_smpl_data(self,*args): return {}
        def forward_animate_gs(self,*args,**kwargs): return {'comp_features':kwargs['features']}
    class Neural(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(3,3).cuda()
        def forward(self,image,features,*args,**kwargs): return self.linear(image+features),None
    renderer=LHMRenderer.__new__(LHMRenderer)
    renderer.model=SimpleNamespace(renderer=Raster(),neural_renderer=Neural(),
                    neural_renderer_input='feats',neural_renderer_patch_size=1)
    cache=torch.ones(1,1,512,512,3,device='cuda',dtype=torch.float16,requires_grad=True)
    args={name:None for name in ['smplx_params','gs_model_list','query_points','render_c2ws','render_intrs','render_bg_colors','motion_emb','pos_emb']}
    args.update(gs_hidden_features=cache,image_latents=cache)
    with pytest.raises(RuntimeError,match='dtype'):
        renderer._render_differentiable(args)
    image,_=renderer.render_differentiable(args)
    image.float().sum().backward()
    assert torch.isfinite(cache.grad).all()

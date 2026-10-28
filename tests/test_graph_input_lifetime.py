import gc
import weakref
from types import SimpleNamespace
import torch
import pytest
from backend.avatar.runtime_cache import RuntimeCache


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_geometry_graph_retains_external_cache_inputs_after_cache_eviction():
    cache=RuntimeCache.__new__(RuntimeCache)
    cache.active=lambda:True;cache.signature=lambda:()
    cache.renderer=SimpleNamespace(state={'parameters':{'transform_mat_neutral_pose':torch.zeros(1,device='cuda')}})
    cache.skinning=SimpleNamespace(weights=torch.ones(3,device='cuda'),joints=torch.ones(3,device='cuda'))
    cache.pose_only=False;cache.graph_error=None;cache.stats={}
    cache.graph=cache.inputs=cache.output=cache.graph_key=cache.graph_shape=None
    attributes=SimpleNamespace(**{name:torch.zeros(1,device='cuda') for name in ['offset_xyz','opacity','rotation','scaling','shs']})
    points=torch.zeros(3,device='cuda')
    cache.original_animate=lambda attrs,pts,data,**kwargs:data['pose']*cache.skinning.weights+cache.skinning.joints
    data={'pose':torch.ones(3,device='cuda'),'betas':torch.zeros(1,device='cuda')}
    with torch.no_grad(): cache.animate(attributes,points,data)
    torch.cuda.synchronize()
    weight=weakref.ref(cache.skinning.weights);joint=weakref.ref(cache.skinning.joints)
    cache.skinning.weights=torch.zeros(3,device='cuda');cache.skinning.joints=torch.zeros(3,device='cuda')
    gc.collect()
    assert weight() is not None and joint() is not None,'Captured tensor inputs were freed by an unrelated cache refresh'
    data['pose'].fill_(3)
    with torch.no_grad(): result=cache.animate(attributes,points,data)
    torch.testing.assert_close(result,torch.full((3,),4.,device='cuda'))

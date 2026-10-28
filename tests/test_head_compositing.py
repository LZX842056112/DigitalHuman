import numpy as np
import torch
from backend.avatar.head_compositing import crop_intrinsics,fit_similarity,composite_head,front_blend_weight


def test_head_roi_uses_exact_body_camera_rays():
    intrinsic=torch.tensor([[1200.,0,384],[0,1200,384],[0,0,1.]])
    points=torch.tensor([[.2,.1,2.],[-.1,-.1,2.5]])
    cropped=crop_intrinsics(intrinsic,210,70,120,512)
    full=points@intrinsic.T;full=full[:,:2]/full[:,2:]
    actual=points@cropped.T;actual=actual[:,:2]/actual[:,2:]
    torch.testing.assert_close(actual,(full-torch.tensor([210,70]))*512/120)
    assert intrinsic[0,2]==384


def test_similarity_preserves_pose_and_uniform_gaussian_scale():
    source=torch.tensor([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,1.],[0.,0.,-1.]])
    rotation=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    target=source@rotation*.2+torch.tensor([.1,.4,.2])
    scale,matrix,translation=fit_similarity(source,target)
    torch.testing.assert_close(source@matrix*scale+translation,target,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(scale,torch.tensor(.2))
    assert torch.det(matrix)>0


def test_head_composite_preserves_body_foreground_and_pixels_outside_head():
    body=torch.full((2,2,3),.2)
    head=torch.zeros_like(body);head[0,0]=.8;head[0,1]=.8
    alpha=torch.tensor([[1.,1.],[0.,0.]])
    old=torch.tensor([[1.,1.],[0.,0.]])
    occluder=torch.tensor([[0.,1.],[0.,0.]])
    result=composite_head(body,head,alpha,old,occluder,torch.ones(2,2))
    torch.testing.assert_close(result[0,0],torch.full((3,),.8))
    torch.testing.assert_close(result[0,1],body[0,1])
    torch.testing.assert_close(result[1],body[1])


def test_single_front_reference_fades_to_original_geometry_at_unseen_angles():
    import math
    assert front_blend_weight(1)==1
    assert front_blend_weight(math.cos(math.radians(30)))==1
    assert 0<front_blend_weight(math.cos(math.radians(65)))<1
    assert front_blend_weight(0)==0
    assert front_blend_weight(-1)==0

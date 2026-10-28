from types import SimpleNamespace
import torch
from backend.avatar.runtime_cache import SkinningCache, token


class Skin:
    def __init__(self):
        self.is_face=torch.tensor([True,False,False])
        self.is_rhand=self.is_lhand=torch.zeros(3,dtype=torch.bool)
        self.expr_dirs=torch.zeros(3,3,1);self.expr_dirs[2,0,0]=1
        self.skinning_weight=torch.tensor([[1.,0.],[.5,.5],[.5,.5]])
        self.base_skinning=SimpleNamespace(joint_num=2)
        self.queries=0;self.joint_calls=0

    def query_voxel_skinning_weights(self,points):
        self.queries+=1
        weight=points[...,0].sigmoid()
        return torch.stack([1-weight,weight],-1)

    def get_transform_mat_vertex(self,transform,points,mask):
        weights=self.query_voxel_skinning_weights(points).clone()
        weights[mask]=self.skinning_weight[None].expand_as(weights)[mask]
        return (weights@transform.reshape(transform.shape[0],2,16)).reshape(*points.shape[:2],4,4)

    def get_zero_pose_human(self,shape,device,face_offset,joint_offset,return_mesh=False):
        self.joint_calls+=1
        return shape.square()


def test_cached_weights_preserve_dynamic_expression_and_invalidate_offset_changes():
    skin=Skin();offset=torch.zeros(3,3)
    cache=SkinningCache(skin,lambda:token(offset))
    points=torch.tensor([[[.1,0,0],[.2,0,0],[.3,0,0]]])
    transform=torch.arange(32,dtype=torch.float32).reshape(1,2,4,4)
    mask=skin.is_face[None]
    with torch.no_grad():
        cache.matrices(transform,points,mask)
        points[:,2,0]+=.4
        actual=cache.matrices(transform+1,points,mask)
        expected=skin.get_transform_mat_vertex(transform+1,points,mask)
        torch.testing.assert_close(actual,expected)
        # A head-position calibration changes the pointer/version signature.
        offset.add_(.1);points[:,1,0]+=.2
        torch.testing.assert_close(cache.matrices(transform,points,mask),
                                   skin.get_transform_mat_vertex(transform,points,mask))
    points=points.clone().requires_grad_(True)
    actual=cache.matrices(transform,points,mask)
    expected=skin.get_transform_mat_vertex(transform,points,mask)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),points)[0],
                               torch.autograd.grad(expected.sum(),points)[0])


def test_joint_cache_supports_cpu_and_changed_body_shape_with_gradients(monkeypatch):
    def no_cuda_runtime():
        raise AssertionError('CPU path must not require a CUDA runtime')
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',no_cuda_runtime)
    skin=Skin();cache=SkinningCache(skin,lambda:None)
    shape=torch.tensor([[1.,2.]])
    with torch.no_grad():
        first=cache.zero_pose(shape,'cpu',None,None)
        assert cache.zero_pose(shape,'cpu',None,None) is first
        assert skin.joint_calls==1
        shape.add_(1)
        torch.testing.assert_close(cache.zero_pose(shape,'cpu',None,None),shape.square())
        assert skin.joint_calls==2
    shape.requires_grad_(True)
    cache.zero_pose(shape,'cpu',None,None).sum().backward()
    torch.testing.assert_close(shape.grad,2*shape.detach())


def test_fixed_head_calibration_keeps_exact_values_and_head_position_gradients():
    skin=Skin();skin.expr_dirs.zero_()
    cache=SkinningCache(skin,lambda:None)
    fixed=skin.is_face;transform=torch.arange(32,dtype=torch.float32).reshape(1,2,4,4)
    base=torch.tensor([[[.1,0,0],[.2,0,0],[.3,0,0]]])
    delta=torch.zeros(1,1,3,requires_grad=True)
    def points():
        value=base.clone();value[:,fixed]=value[:,fixed]+delta
        return value
    def posed(matrices,value):
        homogeneous=torch.nn.functional.pad(value,[0,1],value=1)
        return (matrices@homogeneous[...,None])[...,:3,0]
    reference_points=points()
    reference=posed(skin.get_transform_mat_vertex(transform,reference_points,fixed[None]),reference_points)
    reference_grad=torch.autograd.grad(reference.sum(),delta)[0]
    with cache.fixed_head_calibration(fixed):
        value=points();actual=posed(cache.matrices(transform,value,fixed[None]),value)
        torch.testing.assert_close(actual,reference)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(),delta)[0],reference_grad)
        queries=skin.queries
        with torch.no_grad(): delta.add_(.01)
        value=points();cache.matrices(transform,value,fixed[None])
        assert skin.queries==queries
    assert cache.head_context is None


def test_head_calibration_cache_rejects_trainable_body_points():
    import pytest
    skin=Skin();cache=SkinningCache(skin,lambda:None)
    with pytest.raises(ValueError,match='固定'):
        with cache.fixed_head_calibration(torch.tensor([True,True,False])): pass

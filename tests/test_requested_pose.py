from types import SimpleNamespace
import torch
from backend.avatar.runtime_cache import pose_requested_views


def test_requested_pose_uses_live_angles_and_keeps_canonical_state_unchanged():
    seen=[]
    class Gaussian:
        def __init__(self,**values): self.__dict__.update(values)
    class Raster:
        gs_net=SimpleNamespace(use_rgb=False)
        def _transform_points(self,data,points,offset,device,mesh):
            seen.append(data['body_pose'].shape[0])
            return {'posed_coords':points[None]+offset[None]+data['trans'][:,None],
                    'transform_mat_posed':data['body_pose'],'mesh_meta':mesh}
        def _compute_rotations(self,transform,rotation,device,mesh):
            return rotation[None].expand(transform.shape[0],-1,-1)
    points=torch.randn(5,3);canonical=[Gaussian(xyz=points.clone())]
    attributes=SimpleNamespace(offset_xyz=torch.ones_like(points)*.1,opacity=None,
        rotation=torch.ones(5,4),scaling=None,shs=None)
    data={'body_pose':torch.zeros(1,21,3),'trans':torch.tensor([[.2,0,0]])}
    posed,saved=pose_requested_views(Raster(),attributes,points,data,{},canonical)
    assert seen==[1] and len(posed)==1 and saved is canonical
    torch.testing.assert_close(posed[0].xyz,points+.1+data['trans'])
    torch.testing.assert_close(canonical[0].xyz,points)
    data['trans']+=.3
    moved,_=pose_requested_views(Raster(),attributes,points,data,{},canonical)
    torch.testing.assert_close(moved[0].xyz-posed[0].xyz,torch.ones_like(points)*.3)

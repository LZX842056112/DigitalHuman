import sys
from types import ModuleType,SimpleNamespace
import pytest
import torch
from backend.avatar.runtime_cache import RuntimeCache


@pytest.mark.parametrize('tile_size',[8,16])
def test_raster_partition_keeps_all_points_features_and_camera_resolution(monkeypatch,tile_size):
    captured={};module=ModuleType('core.models.rendering.gsplat_renderer')
    module.scale_intrs=lambda intrinsic,*args:intrinsic
    def rasterization(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1,74,74,128),torch.ones(1,74,74,1),None
    module.rasterization=rasterization
    monkeypatch.setitem(sys.modules,module.__name__,module)
    xyz=torch.arange(9,dtype=torch.float32).reshape(3,3)
    features=torch.arange(3*128,dtype=torch.float32).reshape(3,128)
    cache=RuntimeCache.__new__(RuntimeCache);cache.active=lambda:True;cache.raster_tile_size=tile_size
    cache.raster=SimpleNamespace(background_embedding=torch.zeros(3,128),
        get_gaussians_properties=lambda *args:(xyz,None,None,torch.ones(3,1),torch.ones(3,3),torch.ones(3,4),None))
    camera=SimpleNamespace(width=518,height=518,intrinsic=torch.eye(4),
                           world_view_transform=torch.eye(4),znear=.01,zfar=100)
    result=cache.single_view(None,camera,torch.ones(3),features=features,patch_size=7)
    assert captured['tile_size']==tile_size
    assert captured['width']==captured['height']==74
    assert captured['channel_chunk']==128
    torch.testing.assert_close(captured['means'],xyz)
    torch.testing.assert_close(captured['colors'],features)
    assert result['comp_features'].shape==(74,74,128)

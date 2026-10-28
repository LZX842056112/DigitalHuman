import torch
from backend.avatar.runtime_cache import PositionCache


class Head:
    def __init__(self): self.calls=0
    def _apply_pos_embed(self,x,width,height,ratio=.1):
        self.calls+=1
        y=torch.linspace(-1,1,x.shape[-2],dtype=x.dtype,device=x.device)[:,None]
        xgrid=torch.linspace(-1,1,x.shape[-1],dtype=x.dtype,device=x.device)[None,:]
        position=torch.sin(xgrid*width/height)+torch.cos(y)
        return x+position[None,None]*ratio


def test_position_cache_keeps_values_and_gradients_for_changing_features():
    head=Head();cached=PositionCache(head)
    first=torch.randn(2,4,9,7,requires_grad=True)
    second=torch.randn_like(first,requires_grad=True)
    expected=head._apply_pos_embed(first,512,512)
    actual=cached(first,512,512)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    expected_second=head._apply_pos_embed(second,512,512)
    calls=head.calls
    torch.testing.assert_close(cached(second,512,512),expected_second,rtol=0,atol=0)
    assert head.calls==calls
    actual.sum().backward()
    torch.testing.assert_close(first.grad,torch.ones_like(first))
    # The original input and all learned network weights remain untouched.
    assert len(cached.entries)==1


def test_position_cache_separates_resolution_aspect_ratio_and_precision():
    head=Head();cached=PositionCache(head)
    for width,height,shape,dtype in [(512,512,(1,4,9,7),torch.float32),
        (1024,512,(1,4,9,7),torch.float32),(512,512,(1,4,8,7),torch.float32),
        (512,512,(1,4,9,7),torch.float16)]:
        value=torch.ones(shape,dtype=dtype)
        torch.testing.assert_close(cached(value,width,height),head._apply_pos_embed(value,width,height),rtol=0,atol=0)
    assert len(cached.entries)==4

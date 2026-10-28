import contextlib
import threading
import numpy as np
import torch
from backend.avatar.lhmpp import LHMRenderer


def test_runtime_transfers_uint8_pixels_and_preserves_the_original_display_values(monkeypatch):
    transferred=[]
    class ObservedTensor:
        def __init__(self,value): self.value=value
        def __getitem__(self,key): return ObservedTensor(self.value[key])
        def clamp(self,*args): return ObservedTensor(self.value.clamp(*args))
        def mul(self,value): return ObservedTensor(self.value.mul(value))
        def to(self,dtype): return ObservedTensor(self.value.to(dtype))
        def cpu(self):
            transferred.append(self.value.dtype)
            return self
        def numpy(self): return self.value.numpy()
    values=torch.linspace(-.1,1.1,512*512*3,dtype=torch.float32).reshape(1,1,512,512,3)
    renderer=LHMRenderer.__new__(LHMRenderer)
    renderer.lock=threading.RLock();renderer._closed=False
    renderer.arguments=lambda *args:{}
    renderer._render_differentiable=lambda args:(ObservedTensor(values),None)
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:contextlib.nullcontext())
    image=renderer.render('idle',0,{},np.zeros(100),np.zeros(3))
    assert transferred==[torch.uint8],'The GPU readback still transfers floating point pixels'
    expected=(values[0,0].clamp(0,1).numpy()*255).astype(np.uint8)
    np.testing.assert_array_equal(image,expected)
    assert image.shape==(512,512,3) and image.dtype==np.uint8

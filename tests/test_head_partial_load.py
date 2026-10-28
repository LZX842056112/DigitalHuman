import sys
from types import ModuleType
import pytest
from backend.avatar import head_model,head_refiner


def test_partial_head_cuda_load_offloads_network_even_if_cuda_raises(tmp_path,monkeypatch):
    class Decoder:
        on_gpu=False
        def cuda(self):
            self.on_gpu=True
            raise RuntimeError('CUDA allocation failed')
        def cpu(self):self.on_gpu=False
    decoder=Decoder()
    fake_flame=ModuleType('gagavatar.libs.flame_model');fake_flame.FLAMEModel=object
    monkeypatch.setitem(sys.modules,'gagavatar.libs.flame_model',fake_flame)
    monkeypatch.setattr(head_model,'configure',lambda *_:None)
    monkeypatch.setattr(head_model,'create_decoder',lambda *_:decoder)
    monkeypatch.setattr(head_refiner,'sha256',lambda *_:'flame-hash')
    with pytest.raises(RuntimeError,match='allocation'):
        head_refiner.HeadRefiner(tmp_path,{}, {'flame_sha256':'flame-hash'})
    assert not decoder.on_gpu

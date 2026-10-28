import sys
from types import ModuleType
import torch
import json
from safetensors.torch import save_file


def test_model_loader_initializes_accelerate_before_upstream_logging(tmp_path,monkeypatch):
    import accelerate
    from accelerate.state import PartialState
    from accelerate.logging import get_logger
    from backend.avatar import lhmpp
    monkeypatch.setattr(PartialState,'_shared_state',{})
    # Keep the real state/logger contract; select CPU only for this unit boundary.
    monkeypatch.setattr(accelerate,'PartialState',lambda:PartialState(cpu=True))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(lhmpp,'configure_upstream',lambda _:None)
    weights=tmp_path/'models/lhmpp/model.safetensors'
    weights.parent.mkdir(parents=True)
    save_file({'encoder.weight':torch.ones(1,1)},weights)
    (weights.parent/'config.json').write_text(json.dumps({'neural_renderer':{'type':'test'}}))
    class Model(torch.nn.Module):
        def __init__(self,config):
            super().__init__()
            get_logger('digitalhuman.test.upstream').info('Initializing model')
            self.encoder=torch.nn.Linear(1,1,bias=False)
        def train(self,mode=True):
            super().train(mode)
            # Pinned upstream overrides train() without returning self.
    class Dino:
        _build_dinov2=staticmethod(lambda *args,**kwargs:None)
    for name,attrs in {
        'core.models':{'model_dict':{'human_lrm_a4o':Model}},
        'core.utils.hf_hub':{'wrap_model_hub':lambda cls:cls},
        'core.models.encoders.dinov2_wrapper':{'Dinov2Wrapper':Dino},
    }.items():
        module=ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,module)
    assert isinstance(lhmpp.load_model(tmp_path),Model)

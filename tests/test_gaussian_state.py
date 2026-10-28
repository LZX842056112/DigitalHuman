from dataclasses import dataclass
from types import ModuleType
import sys
import torch


def test_gaussian_output_fields_survive_safe_state_roundtrip(tmp_path, monkeypatch):
    # Complete upstream GaussianAppOutput boundary, without importing CUDA rasterizers.
    @dataclass
    class GaussianAppOutput:
        offset_xyz: object
        opacity: object
        rotation: object
        scaling: object
        shs: object
        use_rgb: bool
    GaussianAppOutput.__module__ = "core.outputs.output"
    module = ModuleType("core.outputs.output")
    module.GaussianAppOutput = GaussianAppOutput
    monkeypatch.setitem(sys.modules, "core.outputs.output", module)
    from backend.avatar.state import save_state, load_state
    gaussians = GaussianAppOutput(torch.tensor([[.1,.2,.3]]), torch.ones(1,1),
                                 torch.tensor([[1.,0.,0.,0.]]), torch.ones(1,3)*.01,
                                 torch.tensor([[[.2,.3,.4]]]), True)
    save_state(tmp_path, {"gs_model_list": [gaussians]})
    restored = load_state(tmp_path)["gs_model_list"][0]
    assert type(restored) is GaussianAppOutput
    assert torch.equal(restored.offset_xyz, torch.tensor([[.1,.2,.3]]))
    assert restored.use_rgb is True

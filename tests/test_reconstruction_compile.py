"""The model adapter must handle DINO's optional latent-grid attribute."""
import json
import sys
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

from backend.avatar import lhmpp


@pytest.fixture
def reconstruction_model(tmp_path, monkeypatch):
    import accelerate
    from accelerate.state import PartialState

    # Replace the multi-gigabyte upstream constructor, but keep the real
    # compiler, module attribute lookup, checkpoint loader and adapter.
    class Encoder(torch.nn.Linear):
        downsample_ratio = 14

        @torch.compile(backend='eager')
        def forward(self, image):
            return super().forward(image)

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = Encoder(1, 1, bias=False)

        @torch.compile(backend='eager')
        def forward_latent_points(self, image):
            _, _, height, width = image.shape
            try:
                grid = self.encoder.patch_size
            except AttributeError:
                ratio = self.encoder.downsample_ratio
                grid = (height // ratio, width // ratio)
            return self.encoder(image.mean().reshape(1, 1)).expand(*grid)

    class Dino:
        _build_dinov2 = staticmethod(lambda *args, **kwargs: None)

    weights = tmp_path / 'models/lhmpp/model.safetensors'
    weights.parent.mkdir(parents=True)
    save_file({'encoder.weight': torch.tensor([[2.]])}, weights)
    weights.with_name('config.json').write_text(json.dumps({
        'encoder_type': 'dinov2', 'neural_renderer': {'type': 'test'},
    }))
    monkeypatch.setattr(PartialState, '_shared_state', {})
    monkeypatch.setattr(accelerate, 'PartialState', lambda: PartialState(cpu=True))
    monkeypatch.setattr(lhmpp, 'configure_upstream', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    for name, attrs in {
        'core.models': {'model_dict': {'human_lrm_a4o': Model}},
        'core.utils.hf_hub': {'wrap_model_hub': lambda cls: cls},
        'core.models.encoders.dinov2_wrapper': {'Dinov2Wrapper': Dino},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    return lhmpp.load_model(tmp_path)


@pytest.mark.parametrize(('height', 'width', 'grid'), [(28, 42, (2, 3)), (42, 56, (3, 4))])
def test_reconstruction_without_patch_size_uses_each_input_grid(reconstruction_model, height, width, grid):
    result = reconstruction_model.forward_latent_points(torch.ones(1, 3, height, width))
    torch.testing.assert_close(result, torch.full(grid, 2.))


def test_reconstruction_keeps_real_model_errors_visible(reconstruction_model):
    # A normal model exception must still reach the caller. In particular,
    # disabling the compiler must not suppress an invalid input rank.
    with pytest.raises(ValueError, match='not enough values to unpack'):
        reconstruction_model.forward_latent_points(torch.ones(3, 28, 42))

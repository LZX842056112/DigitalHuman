import pytest
import torch
from safetensors.torch import save_file
from backend.avatar import lhmpp


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_block=torch.nn.ModuleList([torch.nn.Linear(1,1)])
        self.dense_predictor=torch.nn.Linear(1,1,bias=False)
    def forward(self,value):
        return self.dense_predictor(value)


class PublishedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Linear(1,1,bias=False)
        self.neural_renderer=Decoder()
        self.id_face_net=torch.nn.Linear(1,1,bias=False)


def test_published_dpt_only_checkpoint_loads_all_active_weights_and_keeps_identity(tmp_path):
    model=PublishedModel()
    identity=model.id_face_net
    original_identity=identity.weight.detach().clone()
    path=tmp_path/'model.safetensors'
    save_file({'encoder.weight':torch.tensor([[2.]]),
               'neural_renderer.dense_predictor.weight':torch.tensor([[3.]])},path)
    lhmpp.load_published_checkpoint(model,path,'patch_4dptonly')
    assert model.id_face_net is identity
    assert torch.equal(identity.weight,original_identity)
    assert len(model.neural_renderer.transformer_block)==0
    assert model.neural_renderer(model.encoder(torch.ones(1,1))).item()==6


@pytest.mark.parametrize('renderer_type',['patch_4dptonly','efficientdpt'])
def test_published_checkpoint_never_accepts_missing_active_parameters(tmp_path,renderer_type):
    model=PublishedModel()
    path=tmp_path/'incomplete.safetensors'
    save_file({'neural_renderer.dense_predictor.weight':torch.ones(1,1)},path)
    with pytest.raises(RuntimeError,match='encoder'):
        lhmpp.load_published_checkpoint(model,path,renderer_type)

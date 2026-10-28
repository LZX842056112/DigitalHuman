"""Pinned GAGAvatar inference networks; no training loss or extra encoder download."""
import importlib
import sys
from pathlib import Path
import torch
from backend.storage import sha256

MODEL_SHA256='75d449fd1d67dd7bbc613cef17f0196b1c84766935adc75d5231bc89e33950f8'


def configure(root:Path):
    import tomli
    sys.modules.setdefault('tomllib',tomli)
    repo=root/'third_party/gag_avatar'
    if not (repo/'gagavatar/models/modules/__init__.py').is_file():
        raise FileNotFoundError('GAGAvatar 源码未安装，请运行 scripts/setup_head.ps1')
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))


def load_published_head(root:Path):
    from backend.avatar.lhmpp import configure_upstream
    path=root/'models/gag-head/GAGAvatar.pt'
    if sha256(path)!=MODEL_SHA256:raise ValueError('GAGAvatar 模型校验失败')
    configure(root);configure_upstream(root)
    from core.models.encoders.dinov2.hub.backbones import dinov2_vitb14
    module=importlib.import_module('gagavatar.models.GAGAvatar.models')
    original_hub,original_loss=torch.hub.load,module.FacePerceptualLoss
    def local_dino(repo,name,*args,**kwargs):
        if (repo,name)!=('facebookresearch/dinov2','dinov2_vitb14'):
            raise ValueError('头部编码器类型不匹配')
        return dinov2_vitb14(pretrained=False)
    torch.hub.load=local_dino
    module.FacePerceptualLoss=lambda *args,**kwargs:torch.nn.Identity()
    try:model=module.GAGAvatar()
    finally:torch.hub.load=original_hub;module.FacePerceptualLoss=original_loss
    # Only the hash-verified, published upstream checkpoint uses its legacy
    # pickle container. Saved avatar sidecars always use safetensors.
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    weights=checkpoint.get('model',checkpoint)
    unused=[key for key in weights if key.startswith('percep_loss.')]
    state={key:value for key,value in weights.items() if key not in unused}
    mask_key='base_model.dino_model.mask_token'
    if mask_key in state and mask_key not in model.state_dict():
        unused.append(mask_key);state.pop(mask_key)
    model.load_state_dict(state,strict=True)
    model.eval();model.requires_grad_(False)
    return model,{'strict_loaded_tensors':len(state),'unused_training_tensors':unused,'model_sha256':MODEL_SHA256}


def create_decoder(root:Path,state):
    configure(root)
    from gagavatar.models.modules import StyleUNet
    decoder=StyleUNet(in_size=512,in_dim=32,out_dim=3,out_size=512)
    decoder.load_state_dict(state,strict=True)
    return decoder.eval().requires_grad_(False)

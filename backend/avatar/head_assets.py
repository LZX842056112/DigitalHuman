"""Atomic, data-only head sidecars bound to an unchanged body state."""
from pathlib import Path
import uuid
from backend.storage import atomic_json,read_json,sha256
from backend.avatar.state import require_finite_tensor


def save_head_assets(folder:Path,tensors:dict,metadata:dict):
    from safetensors.torch import save_file
    values={}
    for key,tensor in tensors.items():
        require_finite_tensor(tensor)
        values[key]=tensor.detach().cpu().contiguous().clone()
    body_hash=sha256(folder/'state.json')
    relative=f'.heads/{uuid.uuid4().hex}/head.safetensors'
    target=folder/relative;target.parent.mkdir(parents=True)
    save_file(values,str(target))
    atomic_json(folder/'head.json',{'schema_version':1,'body_state_sha256':body_hash,
        'tensor_file':relative,'tensor_sha256':sha256(target),'metadata':metadata})


def load_head_assets(folder:Path):
    from safetensors.torch import load_file
    schema=read_json(folder/'head.json')
    if schema.get('schema_version')!=1:raise ValueError('头部资产版本不受支持')
    if sha256(folder/'state.json')!=schema.get('body_state_sha256'):
        raise ValueError('头部资产与当前身体状态不匹配，请重新构建头部')
    target=(folder/schema['tensor_file']).resolve()
    if folder.resolve() not in target.parents:raise ValueError('头部张量路径越界')
    if sha256(target)!=schema['tensor_sha256']:raise ValueError('头部张量校验失败')
    tensors=load_file(str(target),device='cpu')
    for tensor in tensors.values():require_finite_tensor(tensor)
    return tensors,schema['metadata']

from pathlib import Path
import uuid
import numpy as np
from backend.storage import atomic_json, read_json, sha256


def require_finite_tensor(tensor):
    import torch
    if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
        raise ValueError('人物状态含无效数值，不能保存或加载')


def encode_tree(value, tensors):
    import torch
    if isinstance(value, (torch.Tensor, np.ndarray)):
        name = f"tensor_{len(tensors):05d}"
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
        require_finite_tensor(tensor)
        tensors[name] = tensor.detach().cpu().contiguous().clone()
        return {"__tensor__": name}
    if type(value).__module__ == "core.outputs.output" and type(value).__name__ == "GaussianAppOutput":
        return {"__gaussian_app__": {name: encode_tree(getattr(value, name), tensors)
                                      for name in ("offset_xyz", "opacity", "rotation", "scaling", "shs", "use_rgb")}}
    if isinstance(value, dict):
        return {"__dict__": {str(k): encode_tree(v, tensors) for k, v in value.items()}}
    if isinstance(value, (tuple, list)):
        return {"__tuple__" if isinstance(value, tuple) else "__list__": [encode_tree(v, tensors) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if type(value).__module__ == "core.structures.gaussian_model" and type(value).__name__ == "GaussianModel":
        return {"__gaussian__": {name: encode_tree(getattr(value, name), tensors)
                                  for name in ("xyz", "opacity", "rotation", "scaling", "shs", "use_rgb")}}
    raise TypeError(f"无法安全保存状态类型: {type(value).__name__}")


def decode_tree(value, tensors):
    if not isinstance(value, dict):
        return value
    if set(value) == {"__tensor__"}:
        return tensors[value["__tensor__"]]
    if set(value) == {"__dict__"}:
        return {k: decode_tree(v, tensors) for k, v in value["__dict__"].items()}
    if set(value) == {"__tuple__"}:
        return tuple(decode_tree(v, tensors) for v in value["__tuple__"])
    if set(value) == {"__list__"}:
        return [decode_tree(v, tensors) for v in value["__list__"]]
    if set(value) == {"__gaussian__"}:
        from core.structures.gaussian_model import GaussianModel
        return GaussianModel(**{k: decode_tree(v, tensors) for k, v in value["__gaussian__"].items()})
    if set(value) == {"__gaussian_app__"}:
        from core.outputs.output import GaussianAppOutput
        return GaussianAppOutput(**{k: decode_tree(v, tensors) for k, v in value["__gaussian_app__"].items()})
    raise ValueError("人物状态包含未知或可执行的对象类型")


def save_state(folder: Path, state):
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    tree = encode_tree(state, tensors)
    relative = f'.states/{uuid.uuid4().hex}/state.safetensors'
    target = folder / relative
    target.parent.mkdir(parents=True)
    save_file(tensors, str(target))
    # The only commit point is this atomic descriptor replacement. Until it
    # succeeds, all readers still see the complete previous version.
    atomic_json(folder / "state.json", {"schema_version": 2, "tree": tree,
                'tensor_file':relative, 'tensor_sha256':sha256(target)})


def _tensor_file(folder, schema):
    name = schema.get('tensor_file','state.safetensors')
    path = (folder/name).resolve()
    if folder.resolve() not in path.parents:
        raise ValueError('人物张量路径越界')
    return name


def state_artifacts(folder):
    return ['state.json', _tensor_file(folder, read_json(folder/'state.json'))]


def load_state(folder: Path, device="cpu"):
    from safetensors.torch import load_file
    schema = read_json(folder / "state.json")
    if schema.get("schema_version") not in {1,2}:
        raise ValueError("人物状态版本不受支持")
    name = _tensor_file(folder, schema)
    if schema.get('tensor_sha256') and sha256(folder/name) != schema['tensor_sha256']:
        raise ValueError('人物张量文件校验失败')
    tensors = load_file(str(folder/name), device=device)
    for tensor in tensors.values():
        require_finite_tensor(tensor)
    return decode_tree(schema["tree"], tensors)

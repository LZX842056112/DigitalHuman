import torch
import pytest
from backend.avatar.head_assets import save_head_assets, load_head_assets


def test_head_sidecar_roundtrip_keeps_body_state_and_rejects_changed_body(tmp_path):
    (tmp_path/'state.json').write_text('{"body":"original"}')
    before=(tmp_path/'state.json').read_bytes()
    save_head_assets(tmp_path,{'xyz':torch.ones(4,3)},{'source_frame':120})
    tensors,metadata=load_head_assets(tmp_path)
    torch.testing.assert_close(tensors['xyz'],torch.ones(4,3))
    assert metadata['source_frame']==120
    assert (tmp_path/'state.json').read_bytes()==before
    (tmp_path/'state.json').write_text('{"body":"changed"}')
    with pytest.raises(ValueError,match='身体'):load_head_assets(tmp_path)


def test_head_sidecar_checks_hash_and_path_before_loading_tensors(tmp_path):
    import json
    (tmp_path/'state.json').write_text('{}')
    save_head_assets(tmp_path,{'xyz':torch.ones(4,3)},{})
    descriptor=json.loads((tmp_path/'head.json').read_text())
    (tmp_path/descriptor['tensor_file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='校验'):load_head_assets(tmp_path)
    descriptor['tensor_file']='../outside.safetensors'
    (tmp_path/'head.json').write_text(json.dumps(descriptor))
    with pytest.raises(ValueError,match='越界'):load_head_assets(tmp_path)


def test_nonfinite_head_assets_cannot_replace_previous_package(tmp_path):
    (tmp_path/'state.json').write_text('{}')
    save_head_assets(tmp_path,{'xyz':torch.ones(4,3)},{})
    before=(tmp_path/'head.json').read_bytes()
    with pytest.raises(ValueError,match='无效'):save_head_assets(tmp_path,{'xyz':torch.tensor([float('nan')])},{})
    assert (tmp_path/'head.json').read_bytes()==before

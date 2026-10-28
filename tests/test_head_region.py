import torch
import pytest


def test_head_region_uses_mesh_semantics_instead_of_vertex_height():
    from backend.avatar.head_region import head_mask
    state={'gs_hidden_features':torch.zeros(1,4,8),'query_points':{'mesh_meta':{'is_face':torch.tensor([False,True,True,False])}}}
    assert torch.where(head_mask(state))[0].tolist()==[1,2]


def test_head_region_rejects_a_mismatched_semantic_map():
    from backend.avatar.head_region import head_mask
    state={'gs_hidden_features':torch.zeros(1,4,8),'query_points':{'mesh_meta':{'is_face':torch.ones(3)}}}
    with pytest.raises(Exception,match='头部'):
        head_mask(state)

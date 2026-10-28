from backend.errors import PipelineError


def head_mask(state):
    import torch
    features=state['gs_hidden_features']
    semantic=state['query_points'].get('mesh_meta',{}).get('is_face')
    if semantic is None:
        raise PipelineError('head_region','人物包缺少头部语义标注，不能安全限定校准区域')
    mask=torch.as_tensor(semantic,device=features.device).reshape(-1).bool()
    if len(mask)!=features.shape[-2] or not mask.any():
        raise PipelineError('head_region','头部语义标注与高斯特征不一致')
    return mask

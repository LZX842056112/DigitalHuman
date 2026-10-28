"""Camera and alpha operations shared by the 3D head/body integration."""
import torch
import math


def front_blend_weight(cosine):
    # A single front reference does not supply the back of the head. Blend
    # continuously into the original full-body 3D reconstruction at 55-80deg.
    low,high=math.cos(math.radians(80)),math.cos(math.radians(55))
    x=max(0.,min(1.,(float(cosine)-low)/(high-low)))
    return x*x*(3-2*x)


def crop_intrinsics(intrinsic,left,top,span,size=512):
    if span<=0 or size<=0:raise ValueError('头部取景尺寸无效')
    result=intrinsic.clone()
    result[0,2]-=left;result[1,2]-=top
    result[:2]*=size/span
    return result


def fit_similarity(source,target):
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3:
        raise ValueError('头部顶点对应关系无效')
    a=source-source.mean(0);b=target-target.mean(0)
    denominator=a.square().sum()
    if not torch.isfinite(denominator) or denominator<=0:raise ValueError('头部形状无效')
    u,s,vh=torch.linalg.svd(a.T@b)
    fix=torch.eye(3,device=source.device,dtype=source.dtype)
    fix[-1,-1]=torch.det(u@vh)
    rotation=u@fix@vh
    scale=(s*torch.diag(fix)).sum()/denominator
    translation=target.mean(0)-source.mean(0)@rotation*scale
    return scale,rotation,translation


def composite_head(body,premultiplied_head,alpha,old_head,body_foreground,head_region):
    visibility=1-body_foreground.clamp(0,1)
    erase=(old_head*head_region*visibility).clamp(0,1)
    background=body*(1-erase[...,None])+erase[...,None]
    coverage=alpha.clamp(0,1)*visibility
    return (premultiplied_head*visibility[...,None]+background*(1-coverage[...,None])).clamp(0,1)

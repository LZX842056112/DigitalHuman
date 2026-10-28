"""Differentiable head anchors tied to SMPL-X and observed image landmarks."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def make_landmark_anchors(points,landmarks,head_mask,neighbors=8):
    head_indices=torch.where(head_mask)[0]
    if not len(head_indices): raise ValueError('缺少头部高斯点')
    distances=torch.cdist(landmarks.float(),points[head_indices].float())
    nearest=distances.topk(min(neighbors,len(head_indices)),largest=False).indices
    indices=head_indices[nearest]
    offset=landmarks.float()-points[indices].float().mean(1)
    return indices,offset


def landmark_projection_loss(points,indices,rest_offset,world_to_camera,intrinsic,observed):
    with torch.autocast(points.device.type,enabled=False):
        anchors=points[indices].float().mean(1)+rest_offset.float()
        camera=F.pad(anchors,[0,1],value=1)@world_to_camera.float().T
        projected=camera[:,:3]@intrinsic.float().T
        uv=projected[:,:2]/projected[:,2:].clamp_min(1e-4)
        return torch.linalg.vector_norm(uv-observed.float(),dim=-1).mean()/512


def training_case(cases,step,balance_photos=False):
    if balance_photos:
        photos=[case for case in cases if case[4].startswith('photo_')]
        videos=[case for case in cases if not case[4].startswith('photo_')]
        if photos and videos:
            group=photos if step%2==0 else videos
            return group[(step//2)%len(group)]
    return cases[step%len(cases)]


def outside_appearance_mask(region):
    outer=F.max_pool2d(region,61,stride=1,padding=30)
    return (outer-region).clamp(0,1)+.2*(1-outer).clamp(0,1)


def outside_appearance_loss(predicted,reference,region,preserved=None):
    """Protect the face boundary and body while allowing inner-face updates."""
    if preserved is None: preserved=outside_appearance_mask(region)
    return ((predicted-reference.detach()).abs()*preserved).sum()/(preserved.sum()*predicted.shape[1]+1e-6)

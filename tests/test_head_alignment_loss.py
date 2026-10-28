import torch
from backend.avatar.head_alignment import make_landmark_anchors,landmark_projection_loss,training_case


def example_geometry():
    points=torch.tensor([[.29,.01,1.],[.31,.01,1.],[.29,-.01,1.],[.31,-.01,1.],[0.,-1.,2.]])
    head=torch.tensor([True,True,True,True,False])
    landmarks=torch.tensor([[.3,0.,1.]])
    world_to_camera=torch.eye(4);world_to_camera[0,3]=-.3
    intrinsic=torch.tensor([[100.,0.,256.],[0.,100.,256.],[0.,0.,1.]])
    return points,head,landmarks,world_to_camera,intrinsic


def test_head_anchors_keep_actual_smpl_landmarks_at_initialization():
    points,head,landmarks,w2c,intrinsic=example_geometry()
    indices,rest=make_landmark_anchors(points,landmarks,head,neighbors=4)
    assert bool(head[indices].all())
    torch.testing.assert_close(points[indices].mean(1)+rest,landmarks)
    target=torch.tensor([[256.,256.]])
    loss=landmark_projection_loss(points,indices,rest,w2c,intrinsic,target)
    assert float(loss)<1e-7


def test_landmark_projection_gradient_moves_head_toward_observed_image_only():
    points,head,landmarks,w2c,intrinsic=example_geometry()
    indices,rest=make_landmark_anchors(points,landmarks,head,neighbors=4)
    delta=torch.zeros_like(points,requires_grad=True)
    target=torch.tensor([[258.,256.]])
    optimizer=torch.optim.SGD([delta],lr=.1)
    before=landmark_projection_loss(points+delta,indices,rest,w2c,intrinsic,target)
    before.backward()
    assert bool(torch.isfinite(delta.grad).all())
    torch.testing.assert_close(delta.grad[~head],torch.zeros_like(delta.grad[~head]))
    optimizer.step()
    assert float(landmark_projection_loss(points+delta,indices,rest,w2c,intrinsic,target))<float(before)


def test_balanced_training_revisits_clear_photo_and_covers_each_video():
    videos=[(None,None,None,None,f'{i:06d}.jpg') for i in range(12)]
    photo=(None,None,None,None,'photo_000.jpg')
    selected=[training_case(videos+[photo],i,True)[4] for i in range(24)]
    assert selected.count('photo_000.jpg')==12
    assert set(selected[1::2])=={v[4] for v in videos}
    assert training_case(videos,13,True) is videos[1]


def test_reference_preservation_penalizes_neck_edits_without_constraining_inner_face():
    from backend.avatar.head_alignment import outside_appearance_loss
    reference=torch.zeros(1,3,32,32,requires_grad=True)
    region=torch.zeros(1,1,32,32);region[:,:,8:24,8:24]=1
    face_only=reference.detach().clone();face_only[:,:,12:20,12:20]=.2
    assert float(outside_appearance_loss(face_only,reference,region))==0
    changed=face_only.clone();changed[:,:,26:30,12:20]=.3;changed.requires_grad_(True)
    loss=outside_appearance_loss(changed,reference,region)
    assert float(loss)>0
    loss.backward()
    assert reference.grad is None,'The preserved image must not be optimized'
    assert not changed.grad[:,:,8:24,8:24].count_nonzero()
    assert changed.grad[:,:,26:30,12:20].abs().sum()>0

import numpy as np
import torch
import pytest
from scipy.spatial.transform import Rotation
from backend.avatar import kinematics


def reference_fk(rotations,joints,parents):
    local=torch.eye(4).repeat(joints.shape[0],joints.shape[1],1,1)
    local[:,:,:3,:3]=rotations
    local[:,0,:3,3]=joints[:,0]
    local[:,1:,:3,3]=joints[:,1:]-joints[:,parents[1:]]
    global_matrices=[local[:,0]]
    for index in range(1,len(parents)):
        global_matrices.append(global_matrices[int(parents[index])]@local[:,index])
    matrices=torch.stack(global_matrices,1)
    rest=torch.cat([joints,torch.zeros_like(joints[...,:1])],-1)[...,None]
    relative=matrices-torch.nn.functional.pad(matrices@rest,[3,0,0,0,0,0])
    return matrices[:,:,:3,3],relative


def test_batched_tree_matches_chain_transforms_and_gradients():
    parents=torch.tensor([-1,0,0,1,1,2,3,5])
    generator=torch.Generator().manual_seed(7)
    joints=torch.randn(2,8,3,generator=generator,requires_grad=True)
    rotations=torch.tensor(Rotation.from_rotvec(np.random.default_rng(4).normal(size=(16,3))).as_matrix(),dtype=torch.float32).reshape(2,8,3,3)
    expected=reference_fk(rotations,joints,parents)
    actual=kinematics.batch_rigid_transform(rotations,joints,parents)
    for value,target in zip(actual,expected): torch.testing.assert_close(value,target,atol=1e-6,rtol=1e-6)
    expected_grad=torch.autograd.grad(expected[1].square().sum(),joints,retain_graph=True)[0]
    actual_grad=torch.autograd.grad(actual[1].square().sum(),joints)[0]
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-5,rtol=1e-5)


def test_quaternion_selection_covers_identity_and_half_turns_without_dynamic_indexing():
    vectors=np.array([[0,0,0],[np.pi,0,0],[0,np.pi,0],[0,0,np.pi],[.3,-1.1,.6]])
    matrices=torch.tensor(Rotation.from_rotvec(vectors).as_matrix(),dtype=torch.float32,requires_grad=True)
    actual=kinematics.matrix_to_quaternion(matrices)
    expected=Rotation.from_rotvec(vectors).as_quat()[:,[3,0,1,2]]
    for value,target in zip(actual.detach().numpy(),expected):
        assert min(np.max(np.abs(value-target)),np.max(np.abs(value+target)))<1e-5
    assert torch.isfinite(actual).all()
    actual.sum().backward()
    assert torch.isfinite(matrices.grad).all()


def test_axis_angle_rotation_keeps_zero_pose_gradients_finite():
    values=torch.tensor([[0.,0.,0.],[.3,-1.1,.6],[0.,0.,3.14159265]],requires_grad=True)
    matrices=kinematics.axis_angle_to_matrix(values)
    expected=torch.tensor(Rotation.from_rotvec(values.detach().numpy()).as_matrix(),dtype=torch.float32)
    torch.testing.assert_close(matrices,expected,atol=1e-6,rtol=1e-6)
    matrices.sum().backward()
    assert torch.isfinite(values.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_captured_kinematics_replays_changed_joint_angles():
    parents=torch.tensor([-1,0,0,1,1,2,3,5])
    angles=torch.zeros(2,8,3,device='cuda')
    joints=torch.arange(48,device='cuda',dtype=torch.float32).reshape(2,8,3)/50
    def forward():
        rotations=kinematics.axis_angle_to_matrix(angles)
        posed,transforms=kinematics.batch_rigid_transform(rotations,joints,parents)
        return posed,kinematics.matrix_to_quaternion(transforms[...,:3,:3])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream),torch.no_grad():
        for _ in range(3): forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph),torch.no_grad():
        captured=forward()
    angles.copy_(torch.linspace(-.3,.5,48,device='cuda').reshape_as(angles))
    graph.replay()
    with torch.no_grad(): expected=forward()
    for actual,target in zip(captured,expected):
        torch.testing.assert_close(actual,target,atol=1e-5,rtol=1e-5)
    assert not torch.allclose(captured[0],joints)

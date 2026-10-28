import torch
from backend.avatar import calibrate


def test_overfit_training_restores_the_best_validation_step():
    parameter=torch.nn.Parameter(torch.tensor([0.]))
    validation=lambda:float((parameter.detach()-.2).square().mean())
    checkpoint=calibrate.ValidationCheckpoint([parameter],validation())
    optimizer=torch.optim.SGD([parameter],lr=.1)
    for step in range(1,11):
        optimizer.zero_grad();(parameter-1).square().mean().backward();optimizer.step()
        checkpoint.consider(validation(),step)
    assert validation()>.4
    assert checkpoint.restore()
    assert checkpoint.step==1 and validation()<1e-10
    torch.testing.assert_close(parameter,torch.tensor([.2]))


def test_no_validation_improvement_restores_baseline_and_is_not_accepted():
    parameter=torch.nn.Parameter(torch.tensor([0.]))
    checkpoint=calibrate.ValidationCheckpoint([parameter],0.)
    optimizer=torch.optim.SGD([parameter],lr=.1)
    for step in range(1,4):
        optimizer.zero_grad();(parameter-1).square().mean().backward();optimizer.step()
        checkpoint.consider(float(parameter.detach().square().mean()),step)
    assert not checkpoint.restore()
    torch.testing.assert_close(parameter,torch.zeros(1))

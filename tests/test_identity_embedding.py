import torch
from backend.avatar import calibrate


def test_arcface_input_uses_published_128_pixel_geometry_and_keeps_input_gradients():
    network=torch.nn.Sequential(torch.nn.Conv2d(1,1,1,bias=False),torch.nn.AvgPool2d(16),
                                torch.nn.Flatten(),torch.nn.Linear(64,2,bias=False))
    with torch.no_grad():
        network[0].weight.fill_(1)
        network[3].weight[0].fill_(1)
        network[3].weight[1,:32]=1
        network[3].weight[1,32:]=-1
    network.eval().requires_grad_(False)
    image=torch.full((1,3,64,64),.75,requires_grad=True)
    embedding=calibrate.identity_embedding(network,image)
    torch.testing.assert_close(embedding,torch.tensor([[1.,0.]]))
    embedding[:,1].sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum()>0

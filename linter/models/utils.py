import torch

def normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)
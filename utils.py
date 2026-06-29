import torch
import torch.nn.functional as F
from torch.nn.init import orthogonal_

def easy_orthogonal(z_dim, total_p):
    I = torch.eye(z_dim)
    W = I.repeat(1, total_p // z_dim + 1)
    W = W[:, :total_p]
    return W

def hard_orthogonal(z_dim, total_p):
    I = torch.empty(z_dim, z_dim)
    orthogonal_(I)
    W = I.repeat(1, total_p // z_dim + 1)
    W = W[:, :total_p]
    return W

def harder_orthogonal(z_dim, total_p):
    blocks = []
    Q = torch.empty(z_dim, z_dim)
    orthogonal_(Q)
    current = torch.eye(z_dim, z_dim)
    num_blocks = total_p//z_dim + 1
    for _ in range(num_blocks):
        blocks.append(current.clone())
        current = current @ Q
    W = torch.cat(blocks, dim=1)
    W = W[:,:total_p]
    W = F.normalize(W, dim=1)
    return  W

def direct_orthogonal(z_dim, total_p):
    W = torch.empty(z_dim, total_p)
    orthogonal_(W)
    return W

def check_nan(name, x):

    if torch.isnan(x).any():
        raise RuntimeError(f"{name} contains NaN")

    if torch.isinf(x).any():
        raise RuntimeError(f"{name} contains NaN")
import torch
import torch
from torch_geometric.loader import DataLoader

from torch.utils.data import DataLoader, TensorDataset

from torch.nn import Sequential, Linear, ReLU
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from pathlib import Path
from torchvision import transforms
from .data_registry import register_dataset
@register_dataset("pointManifold")
class PointManifold(torch.utils.data.Dataset):
    #TODO: This should take the config instead. 
    def __init__(self, root = "data/pointManifold", train = True, transform = None, amount = 2000, d= 100, dim= 3):
        self.train = train
        #add transform capability
        self.transform = transform
        self.root = Path(root)
        self.dataAmt = amount
        self.d = d
        self.csv_path = self.root/"data.csv"
        # TODO: dataset scheduler? 
        self.dim = dim
        #TODO: temporary - weakness is it needs everything in memory at once. Make something nicer later. 
        self.dataset = make_dataset(self.dataAmt, self.d, self.dim, "cpu")
        #TODO: Self.transform do later. 
    def __len__(self):
        return self.dataAmt
    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        return {"inputs": item[0], "outputs": item[1]}
        

    def get_metadata(self):

        return {
            "d": self.d,
            "dim": self.dim, 
        }
def _vmf_sample_general(mu: Tensor, kappa: Tensor, n: int) -> Tensor:
    """
    Batched Wood/Ulrich sampler for S^{d-1}, d>=2.
    mu: (B,d) (need not be unit)
    kappa: (B,) or scalar tensor (>=0)
    Returns: (B,n,d)
    """
    B, d = mu.shape
    device, dtype = mu.device, mu.dtype
    mu = mu / mu.norm(dim=-1, keepdim=True)
    kappa = kappa.to(device=device, dtype=dtype).reshape(-1)  # (B,) or (1,)
    
    # Uniform shortcut
    if (kappa <= 1e-12).all():
        X0 = _uniform_sphere(B, d, n, device, dtype)
        Q = _householder_Q(mu)  # (B,d,d)
        return torch.einsum('bij,bnj->bni', Q, X0)
    if kappa.shape == (1,):
        kappa = kappa[0]*torch.ones((B))
    # Precompute Wood constants per batch
    p = d
    a = (p - 1) / 2.0
    # b = (-2k + sqrt(4k^2 + (p-1)^2)) / (p-1)
    b = (-2.0 * kappa + torch.sqrt(4.0 * kappa * kappa + (p - 1) ** 2)) / (p - 1)
    
    x0 = (1.0 - b) / (1.0 + b)                     # (B,)
    c = kappa * x0 + (p - 1) * torch.log1p(-x0 * x0)  # (B,)

    # Accept–reject for w; vectorized over (B, n), loops only over rejections
    w = torch.empty(B, n, device=device, dtype=dtype)
    done = torch.zeros(B, n, dtype=torch.bool, device=device)
    tiny = torch.finfo(dtype).tiny
    while not done.all():
        need = (~done).sum().item()
        # Sample Z ~ Beta(a,a) via two gammas; generate a fresh pool and fill where needed
        z1 = torch.distributions.Gamma(torch.tensor([a], device=device, dtype=dtype), torch.tensor([1.0], device=device, dtype=dtype)).sample((need,)).squeeze(-1)
        z2 = torch.distributions.Gamma(torch.tensor([a], device=device, dtype=dtype), torch.tensor([1.0], device=device, dtype=dtype)).sample((need,)).squeeze(-1)
        Z = z1 / (z1 + z2 + tiny)  # (need,)
        # Map back to (B,n) indices
        idx = torch.nonzero(~done, as_tuple=False)
        b_sel = b[idx[:, 0]]      # (need,)
        x0_sel = x0[idx[:, 0]]
        k_sel = kappa[idx[:, 0]]
        c_sel = c[idx[:, 0]]

        W = (1 - (1 + b_sel) * Z) / (1 - (1 - b_sel) * Z)  # (need,)
        U = torch.rand(need, device=device, dtype=dtype).clamp_min(tiny)
        accept = (k_sel * W + (p - 1) * torch.log1p(-x0_sel * W) - c_sel) >= torch.log(U)
        # Write accepted
        if accept.any():
            put_idx = idx[accept]
            w[put_idx[:, 0], put_idx[:, 1]] = W[accept]
            done[put_idx[:, 0], put_idx[:, 1]] = True

    # Sample V ~ Unif(S^{p-2}) for each (B,n)
    V = torch.randn(B, n, p - 1, device=device, dtype=dtype)
    V = V / V.norm(dim=-1, keepdim=True).clamp_min(tiny)
    s = torch.sqrt((1 - w * w).clamp_min(0))  # (B,n)
    x0 = torch.cat([V * s.unsqueeze(-1), w.unsqueeze(-1)], dim=-1)  # (B,n,p)

    # Rotate north pole to mu
    Q = _householder_Q(mu)          # (B,d,d)
    X = torch.einsum('bij,bnj->bni', Q, x0)
    return X
def f(x): 
    #x is (B, N, 3) 
    p = torch.ones_like(x)
    t = 2
    c = 10
    return c*torch.exp(-1*((x-p)*(x-p)).sum(dim = -1)/t)
def compute_avg_f(x):
    fx = f(x) #(B, N)
    empirical = fx.mean(-1)
    return empirical

def make_dataset(B, N, d, device):
    z= torch.normal(0, 1, size = (B, d))
    data = z/torch.norm(z, dim = -1)[..., None]
    assert(data.shape == (B, d)) 
    kappa = torch.tensor(10.0, device = device)
    data = _vmf_sample_general(z, kappa, N) # (B, N, d)
    vals = compute_avg_f(data)
    
    dataset = TensorDataset(data, vals)
    return dataset
def _uniform_sphere(dim, n=1):
    """Uniform on S^{dim-1} in R^{dim}."""
    z = np.random.normal(size=(n, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z

@torch.no_grad()
def _householder_Q(mu: Tensor) -> Tensor:
    """
    Batched Householder reflections mapping e_d -> mu.
    mu: (B,d) unit (will be normalized).
    Returns Q: (B,d,d) orthogonal, with Q @ e_d = mu.
    """
    mu = mu / mu.norm(dim=-1, keepdim=True)
    B, d = mu.shape
    e_d = torch.zeros(B, d, device=mu.device, dtype=mu.dtype)
    e_d[:, -1] = 1.0
    diff = e_d - mu
    nrm = diff.norm(dim=-1, keepdim=True)
    # If already aligned (or nearly), return identity
    Q = torch.eye(d, device=mu.device, dtype=mu.dtype).expand(B, d, d).clone()
    mask = (nrm.squeeze(-1) >= 1e-12)
    if mask.any():
        v = torch.zeros_like(diff)
        v[mask] = diff[mask] / nrm[mask]
        # Q = I - 2 v v^T
        Qv = torch.einsum('bi,bj->bij', v, v)
        Q_new = torch.eye(d, device=mu.device, dtype=mu.dtype) - 2.0 * Qv
        Q[mask] = Q_new[mask]
    return Q
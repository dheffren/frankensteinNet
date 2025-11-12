import torch
import torch
from torch_geometric.loader import DataLoader

from torch.utils.data import DataLoader, TensorDataset
from pointNet import PointNet
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
    def __init__(self, root = "data/pointManifold", train = True, transform = None, amount = 2000, d= 100):
        self.train = train
        
        self.transform = transform
        self.root = Path(root)
        self.dataAmt = amount
        self.dataD = d
        # dataset scheduler? 
@register_dataset("ballRotating")
class BallRotatingDataset(torch.utils.data.Dataset):
    #TODO: I have a feeling the root should just be "data/" to be consistent with MNIST, then it should add its own name here. 
    def __init__(self, root = "data/ballRotating", train=True, transform = None, BW = False, img_size= 128, amount = 2000):
        self.train = train
        
        self.transform = transform
        self.root = Path(root)
        self.BW = BW
        self.input_channels = 3
        self.img_size = img_size
        if self.BW: 
            self.input_channels = 1
        self.dataAmt = amount
        self.data_shape = (self.input_channels, img_size, img_size)
       
        self.csv_path = self.root/"data.csv"
       
        #TODO: Make it so don't have to generate data every time. 
        self.create_data(output_csv_path = self.csv_path)
        
        self.entries = pd.read_csv(self.csv_path)
    def __len__(self):
        return self.dataAmt

    def __getitem__(self, idx):
      
        #TODO: Add caching or lazy loading or something faster. 
        row = self.entries.iloc[idx]
        x1 = Image.open(row["path1"])
        x2 = Image.open(row["path2"])
        #return data as a dict. 
        element = {"x1": x1, "x2":x2}
        #includes normalization. 

        if self.transform:
      
            transformedEle = self.transform(element)
            return transformedEle
        else: 
            #still convert PIL image to tensor. 
            transform = transforms.Compose([transforms.ToTensor()])
             
            element = {"x1": transform(x1), "x2":transform(x2)}
            return element
    def get_metadata(self):

        return {
            "input_channels": self.input_channels, 
            "input_shape": self.data_shape, 
            "latent_dimU": 2, 
            "latent_dimV": 2, 
            "latent_dimC": 2
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

def make_dataset(B, N, d, device):
    z= torch.normal(0, 1, size = (B, d))
    data = z/torch.norm(z, dim = -1)[..., None]
    assert(data.shape == (B, d)) 
    kappa = torch.tensor(10.0, device = device)
    data = _vmf_sample_general(z, kappa, N) # (B, N, d)
    vals = compute_avg_f(data)
    show_point_values_real(data[0], labels = vals[0])
    dataset = TensorDataset(data, vals)
    return dataset
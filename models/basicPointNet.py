from baseModel import BaseModel
from torch import Tensor
import torch
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import MessagePassing

from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv import PointNetConv

class PointNet(BaseModel):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #layer norm = normalizing EACH POINT IN CLOUD - not normalizing over clouds. Use instanceNorm instead? 
        self.mlp1 = Sequential(
            Linear(in_channels + 3, 32), torch.nn.LayerNorm(32),
            ReLU(),
            Linear(32, 64),torch.nn.LayerNorm(64), ReLU(), 
            Linear(64, 64),torch.nn.LayerNorm(64), ReLU(), 
            Linear(64, 128))

        self.aggregation = Sequential(
            Linear(128, 64),
            torch.nn.LayerNorm(64),
            ReLU(),
            Linear(64, 32), torch.nn.LayerNorm(32), ReLU(), 
            Linear(32, 2))
    def forward(self, batch):
        h = self.mlp1(batch)
        h_feat = h.mean(dim=1)
        output = self.aggregation(h_feat)
        return {"output": output, "mean":h_feat,  "latents": h}
    def prepare_input(self, batch):
        return {"inputs": batch[0], "outputs": batch[1]}
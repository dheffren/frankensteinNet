from .baseModel import BaseModel
from torch import Tensor
import torch
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import MessagePassing

from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv import PointNetConv
from .registry import register_model
@register_model("PointNet")
class PointNet(BaseModel):
    def __init__(self, model_cfg, loss_fn, hyp_sched,  metadata, device = "cpu"):
        super().__init__(model_cfg, loss_fn, hyp_sched, metadata, device)
        self.device = device
        #layer norm = normalizing EACH POINT IN CLOUD - not normalizing over clouds. Use instanceNorm instead? 
        in_channels = model_cfg.get("in_channels")
        self.mlp1 = Sequential(
            Linear(in_channels + 3, 32), ReLU(),
            
            Linear(32, 64), ReLU(), 
            Linear(64, 64), ReLU(), 
            Linear(64, 128))

        self.aggregation = Sequential(
            Linear(128, 64),
            ReLU(),
            Linear(64, 32), torch.nn.LayerNorm(32), ReLU(), 
            Linear(32, 1))
    def forward(self, batch):
        print(batch.shape)
        h = self.mlp1(batch)
        h_feat = h.mean(dim=1)
        output = self.aggregation(h_feat)
        #TODO: Should output shape be B, 1 or B, ? I think B, 1? Flattening made the test loss go down. 
        output = output.flatten() # Don't like this. 
        return {"output": output, "mean":h_feat,  "latents": h}
    def prepare_input(self, batch, requires_grad = False):
        return {"batch": batch["inputs"].to(self.device)}, {"target": batch["outputs"].to(self.device)}
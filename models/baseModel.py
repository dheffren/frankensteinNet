import torch
import torch.nn as nn
#TODO: Somehow implement this to "force" all models to follow somewhat of the same structure. 
class BaseModel(nn.Module):
    def __init__(self, model_cfg, loss_fn, hyp_sched,  metadata, device = "cpu", track_grad = True):
        return
    

    def compute_loss(self, batch, epoch)
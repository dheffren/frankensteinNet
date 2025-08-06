import torch
import torch.nn as nn
from  abc import ABC, abstractmethod
#TODO: Somehow implement this to "force" all models to follow somewhat of the same structure. 
class BaseModel(nn.Module, ABC):
    def __init__(self, model_cfg, loss_fn, hyp_sched,  metadata, device = "cpu", track_grad = True):
        super().__init__()
       
        self.loss_fn = loss_fn
        self.hyp_sched = hyp_sched
        self.device = device
        
        self.track_grad = track_grad
        #added this
        self.to(self.device)
        return
    @abstractmethod
    def forward(self, **kwargs):
        """
        Outputs: a dict of dicts, with recon, latent as outer ones. 
        """
        pass
    def compute_loss(self, batch, epoch):
        """
        batch = directly obtained from the dataloader. Epoch is a number. 
        call prepare input to get the two dicts. 
        """
        inputs, targets = self.prepare_input(batch)
        out = self(**inputs)
        #this alters the hyperparameters of the loss function. 
        return self.loss_fn(out, targets,**self.hyp_sched.get_all(epoch))
    def compute_loss_helper(self, out, targets, epoch):
        # use if already computed out and targets. 
        return self.loss_fn(out, targets,**self.hyp_sched.get_all(epoch))
    @abstractmethod
    def prepare_input(self, batch):
        """
        Call to prepare input. 
        input: batch (Dict, tensor or list)
        Returns inputs (Dict), targets (Dict)
        Inputs = input to forward method. 
        targets = anything else. 
        """
        pass
    def get_loss(self):
        return self.loss_fn
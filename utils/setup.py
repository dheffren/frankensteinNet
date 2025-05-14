from models.autoencoder import Autoencoder
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from scheduler import ScalarSchedule, SchedBundle
from data import get_dataloaders
from losses import make_loss_fn
import torch

def build_model(config):
    #load the loss function from the config.loss function. 
    model_cfg = config["model"]

    loss_fn = make_loss_fn(config["loss"])
    hyp_scheduler = build_hyp_scheduler(config)
    #TODO: What to do if don't input these things. 
    if model_cfg["type"] == "Autoencoder":
        return Autoencoder(
            model_cfg = model_cfg,
            loss_fn = loss_fn, 
            hyp_sched = hyp_scheduler, 
            device = config["device"]
        )
    #ADD other model types here. 

    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

def build_optimizer(model, config):

    opt_cfg = config["optimizer"]
    if opt_cfg["type"] == "Adam":
        return Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            #should weight decay be here. 
            weight_decay=config["training"].get("weight_decay", 0.0)
        )
    #add more optimizers here. 
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer']['type']}")
    
def build_scheduler(optimizer, config):
    #it's ok to not have a scheduler specified. 
    sched_cfg = config.get("scheduler", {})
    #if no schduler return true. If scheduler but not enabled return true. if scheduler and enabled return false. 
    if not sched_cfg.get("{enabled", False):
        #does nothing. 
        return IdentityScheduler()
    sched_type = sched_cfg["type"]
    if sched_type == "step":
        return StepLR(
            optimizer, 
            step_size = sched_cfg["step_size"], #need step size
            gamma = sched_cfg["gamma"]
        )
    elif sched_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            #TODO: Fix the problem with these parameters not being good. 
            T_max=sched_cfg["T_max"]
        )
    # add more schedulers here. 
    else: 
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
def build_hyp_scheduler(config):
    return SchedBundle(config.get("scheduler_hyp", {}))

def build_dataloaders(config):
    #TODO: Fix get_dataloaders, pick what parameters I want passed into this. Fix the path vs dataset name problem. 
    data_cfg = config["data"]
    return get_dataloaders(
        #path = data_cfg["path"], 
        #dataset_name=config["data"]["name"],
        #batch_size=config["training"]["batch_size"],
        #num_workers=config["data"].get("num_workers", 8), 
        config
    )
class IdentityScheduler:
    def step(self):
        pass
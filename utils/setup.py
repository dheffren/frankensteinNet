from models.autoencoder import Autoencoder
from models.dualConvAutoencoder import ConvolutionalAutoencoder
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from scheduler import ScalarSchedule, SchedBundle
from runManager import RunManager
from torch.utils.data import DataLoader
from logger import Logger

from data import get_dataloaders
from losses import make_loss_fn
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ExperimentBundle:
    model: nn.Module
    #loss_fn: Callable
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any]
    #hyp_scheduler: SchedBundle
    dataloaders: Dict[str, DataLoader]
    logger: Logger
    run_manager: RunManager
    metadata: dict
def setup_experiment(config) -> ExperimentBundle:
    device = config["device"]
    dataloaders, meta = build_dataloaders(config)

    
    model = build_model(config, meta).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    #TODO: Deal with resume 
    run_manager = RunManager(config, "runs", False)
    #don't love this reference here. 
    #TODO: Pass the metadata into logger, and deal with hyperparameter schedule in logger
    logger = Logger(run_manager.run_dir, config, meta)
    return ExperimentBundle(model, optimizer, scheduler, dataloaders, logger, run_manager, meta)

def build_model(config, metadata):
    #load the loss function from the config.loss function. 
    #TODO: DO i want to return the loss function and hyp scheduler?
    model_cfg = config["model"]

    loss_fn = make_loss_fn(config["loss"])
    hyp_scheduler = build_hyp_scheduler(config)
    #TODO: What to do if don't input these things. 
    if model_cfg["type"] == "Autoencoder":
        model = Autoencoder
    #ADD other model types here. 
    elif model_cfg["type"] == "DualConvAutoencoder":
        model = ConvolutionalAutoencoder
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    return model(model_cfg = model_cfg,
            loss_fn = loss_fn, 
            hyp_sched = hyp_scheduler, 
            metadata = metadata, 
            device = config["device"],
            )
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
    return get_dataloaders(
        config
    )
class IdentityScheduler:
    def step(self):
        pass
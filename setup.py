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
from utils.setup_hooks import register_diagnostics_as_hooks, register_hooks_from_config
from hookManager import HookManager
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
    hook_manager: HookManager
    metadata: dict
def setup_experiment(config) -> ExperimentBundle:
    device = config["device"]
    dataloaders, meta = build_dataloaders(config)

    
    model = build_model(config, meta).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    hook_mgr = HookManager()
    register_hooks_from_config(hook_mgr, config)
    register_diagnostics_as_hooks(hook_mgr, config)
    #TODO: Deal with resume 
    run_manager = RunManager(config, "runs", False)
    #don't love this reference here. 
    #TODO: Pass the metadata into logger, and deal with hyperparameter schedule in logger
    logger = Logger(run_manager.run_dir, config, meta)
    return ExperimentBundle(model, optimizer, scheduler, dataloaders, logger, run_manager, hook_mgr, meta)

def build_model(config, metadata):
    #load the loss function from the config.loss function. 
    #TODO: DO i want to return the loss function and hyp scheduler?
    model_cfg = config["model"]

    loss_fn = make_loss_fn(config["loss"])
    hyp_scheduler = build_hyp_scheduler(config)
    #TODO: What to do if don't input these things. 
    from models.registry import get_registered_model
    modelType = get_registered_model(model_cfg["type"])
    print(modelType)
    return modelType(model_cfg, loss_fn, hyp_scheduler, metadata, device = config["device"])

def build_optimizer(model, config):
    #TODO: Make registry. 
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
    #TODO: Make registry. 
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
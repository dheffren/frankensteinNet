from .registry import register_diagnostic
import torch
from utils.hookHelpers import Trigger, StepCtx, Services
#TODO: Make it so can use same method for epoch and step? OR waste of time? Would need to rework hooks as well. 
"""
@register_diagnostic("epoch",default_trigger = "epoch", default_every = 1)
def log_epoch(name, trigger, step,  logger, epoch,  **kwargs):
       # print("logging ep")
        logger.log_scalar(f"{name}", epoch, step=step)
@register_diagnostic("train_metrics", default_trigger = "epoch", default_every = 1)
def log_train_metrics(name, trigger, step, logger,  train_metrics,  **kwargs):
  
    for namee, dict in train_metrics.items():
        for k,v in dict.items():
            if torch.is_tensor(v):
                    v = v.item()
            logger.log_scalar(f"{name}/{trigger}/{namee}/{k}", v, step)
@register_diagnostic("val_metrics", default_trigger = "epoch", default_every = 1)
def log_val_metrics_epoch(name, trigger, step, logger, val_metrics, **kwargs):
    for namee, dict in val_metrics.items():
        for k, v in dict.items():
            if torch.is_tensor(v):
                    v = v.item()
            #print(f"val/{name}/{k}")
            logger.log_scalar(f"{name}/{trigger}/{namee}/{k}", v, step)
@register_diagnostic("checkpoints", default_trigger = "epoch", default_every = 10)
def save_checkpoints(name, trigger, step, model,   logger, epoch, **kwargs):
    #print("saving check")
    logger.save_checkpoint(model, epoch) #this epoch isn't a problem.
@register_diagnostic("learning_rate", default_trigger = "epoch", default_every = 1)
def log_learning_rate(name, trigger, step, logger, lr,**kwargs):
    logger.log_scalar("lr", lr, step)
"""


@register_diagnostic("grad_norm", Trigger. BEFORE_OPT_STEP, default_every =1, priority = 1)
def grad_norm(ctx: StepCtx, services: Services):
    grad_norms = {}
    total_norm_sq = 0.0
    model = services.model
    for name, param in model.named_parameters():
        
        if param.grad is not None:
            #detach grad from everything. Have gradient bc backprop. 
            norm = param.grad.detach().norm(2).item()
    
           # grad_norms["grad_norm/" + name] = norm
            total_norm_sq += norm ** 2
    #hopefully no shared names
    grad_norms["grad_norm/total"] = total_norm_sq**0.5
    """
    if group_layers:
        layer_norms = defaultdict(list)
        for name, norm in grad_norms.items():
            
            layer_name = name.split('.')[0]  # You can customize this grouping rule
        
            layer_norms[layer_name].append(norm ** 2)
        
        for layer, norm_sq_list in layer_norms.items():
            grad_norms[layer] = sum(norm_sq_list) ** 0.5
    """
    return {"metrics": grad_norms}

@register_diagnostic("weight_norm", Trigger. BEFORE_OPT_STEP, default_every =1, priority = 1)
def weight_norm(ctx: StepCtx, services: Services):
    """
    Returns a dict of L2 norms of all model weights.
    """
    model = services.model
    norms = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            norms[f"weight_norm/{name}"] = param.data.norm(2).item()
    #add per layer here as well. 
    """
    if group_layers: 
        layer_norms = defaultdict(list)
        for name, norm in norms.items():
            layer_name = name.split('.')[0]
            layer_norms[layer_name].append(norm**2)
        for layer, norm_sq_list in layer_norms.items():
            norms[layer] = sum(norm_sq_list)**.5
    """
    return {"metrics": norms}

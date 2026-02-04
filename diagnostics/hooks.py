from .registry import register_diagnostic
import torch
from utils.hookHelpers import Trigger, StepCtx, Services
@register_diagnostic("checkpoints", Trigger.EPOCH_END, default_every = 10, priority = 1)
def save_checkpoint(ctx:StepCtx, services: Services):
    #incredibly scuffed way of doing this. 
    #TODO: See if context stuff is right. 
    services.checkpoint()
@register_diagnostic("learning_rate", Trigger.EPOCH_END, default_every = 1, priority = 1)
def log_learning_rate(ctx: StepCtx, services: Services):
    return {"metrics": {"lr": ctx.lr}}



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

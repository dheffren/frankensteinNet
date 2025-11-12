from .registry import register_diagnostic
from utils.hookHelpers import Trigger, StepCtx, Services
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
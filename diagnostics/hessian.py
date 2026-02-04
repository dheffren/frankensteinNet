
from hessian_eigenthings import compute_hessian_eigenthings
from utils.flatten import flatten
from utils.fixedBatch import get_fixed_batch
from .metrics import lanczos, hvp
from .registry import register_diagnostic

import torch
from utils.hookHelpers import * 
@register_diagnostic("hessian", default_trigger = Trigger.EVAL_END, default_every = 5, priority = 0) 
def hessian(ctx: StepCtx, S: Services):
    """Computes PCA over the latent vectors in the model output and logs explained variance ratios.
    Optionally logs a 2D PCA scatter plot.

    Config options (in cfg["diagnostics_config"]):
        - latent_pca_key (str): name of latent vector in model output dict (e.g., "latent")
        - latent_pca_components (int): number of PCA components (default: 5)
        - max_batches (int): max val batches to use
        - plot (bool): whether to generate a 2D PCA scatter plot (default: True)
"""
    #TODO: Add per label details here. 
    model = S.model
    device = S.device
    val_loader = S.val_loader
    epoch = ctx.epoch
    cfg = S.cfg

    diag_cfg = cfg.get("diagnostics_config", {})
    layers = diag_cfg.get("layer_pca_layers", ["latent"])
    n_components = diag_cfg.get("layer_pca_components", 5)
    max_batches = diag_cfg.get("max_batches", 1)
    save_latents = diag_cfg.get("save_latents", False)
    direction_types = diag_cfg.get("direction_types", ['random', 'gradient'])
    num_dirs = diag_cfg.get("num_dirs", 4)
    #heatmap_dirs = 
    epsilons = diag_cfg.get("epsilons", [1e-4, 1e-3, 1e-2])
    num_latents = diag_cfg.get("num_latents", 20)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    model.eval()

    outputDict = {
    }
    params = [p for p in model.parameters() if p.requires_grad]

    dim = sum(p.numel() for p in params)
   
    batch = get_fixed_batch(val_loader, seed, num_samples = num_latents)
    Hv_op = Hv_op_factory(model, batch, epoch)
    #TODO: Set up m as a hyperparameter. 
    eigenvals = lanczos(Hv_op, dim, m = 30, device = cfg["device"], seed = seed)
    
    k = min(5, len(eigenvals))
    topk = torch.topk(eigenvals, k).values
    for i in range(k):
        outputDict[f"eigenval_{k-i}"] = topk[i]
    dictOutput = {
        "metrics": outputDict, 
        "artifacts": []
    }
    return dictOutput

def Hv_op_factory(model, batch, epoch):
    def Hv_op(v):
        return hvp(model, batch, v, epoch, create_graph = False)
    return Hv_op


import torch

from .registry import register_diagnostic
from .helper import * 
@register_diagnostic("latent_norms") 
def latent_norms(model, val_loader, logger, epoch, cfg, meta, **kwargs):
    """
    Computes PCA over the latent vectors in the model output and logs explained variance ratios.
    Optionally logs a 2D PCA scatter plot.

    Config options (in cfg["diagnostics_config"]):
        - latent_pca_key (str): name of latent vector in model output dict (e.g., "latent")
        - latent_pca_components (int): number of PCA components (default: 5)
        - max_batches (int): max val batches to use
        - plot (bool): whether to generate a 2D PCA scatter plot (default: True)

    #TODO: Add per label details here. 
    """
    diag_cfg = cfg.get("diagnostics_config", {})
    layers = diag_cfg.get("layer_pca_layers", ["latent"])
    n_components = diag_cfg.get("layer_pca_components", 5)
    max_batches = diag_cfg.get("max_batches", 1)
    save_latents = diag_cfg.get("save_latents", False)
    num_latents = diag_cfg.get("num_latents", 20)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    model.eval()
    
    outputDict = {
    }
    for layer in layers:
        latents, _ = compute_latent_batch(model, val_loader, layer, seed, num_latents)
        if save_latents: 
            logger.save_artifact(latents.detach().cpu().numpy(), f"{layer}/embed_epoch_{epoch}")
        norms = latents.norm(dim=1)
        output_dict = {f"{layer}/norm_mean": norms.mean().item(),
        f"{layer}/norm_std": norms.std().item(),
        f"{layer}/norm_max": norms.max().item()}
        outputDict.update(output_dict)
        
    return outputDict

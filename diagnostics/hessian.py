
from hessian_eigenthings import compute_hessian_eigenthings
from utils.flatten import flatten
from utils.fixedBatch import get_fixed_batch
from .registry import register_diagnostic
@register_diagnostic() 
def hessian(model, val_loader, logger, epoch, cfg, meta, **kwargs):
    """Computes PCA over the latent vectors in the model output and logs explained variance ratios.
    Optionally logs a 2D PCA scatter plot.

    Config options (in cfg["diagnostics_config"]):
        - latent_pca_key (str): name of latent vector in model output dict (e.g., "latent")
        - latent_pca_components (int): number of PCA components (default: 5)
        - max_batches (int): max val batches to use
        - plot (bool): whether to generate a 2D PCA scatter plot (default: True)
"""
    #TODO: Add per label details here. 
    
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
    num_eigenthings = 20
    print("here")
    print(val_loader)
    eigenvals, eigenvecs = compute_hessian_eigenthings(model, val_loader, None, num_eigenthings)
    print("Done")
    outputDict['eigenval'] = eigenvals
    outputDict['eigenvec'] = eigenvecs
    return outputDict
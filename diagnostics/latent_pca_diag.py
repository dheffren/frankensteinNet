import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .registry import register_diagnostic # Your decorator
import PIL.Image
from visualization import plot_pca_scree, plot_pca_component, plot_pca_2d_scatter, plot_pca_3d_scatter
import io
from utils.flatten import flatten
from utils.fixedBatch import get_fixed_batch
from .helper import *
def pca_field_fn(cfg: dict) -> list[str]:
    dcfg = cfg.get("diagnostics_config", {})
    keys     = dcfg.get("layer_pca_layers", ["latent"])
    fields   = dcfg.get("layer_pca_fields", [])          # may be empty
    comps    = dcfg.get("layer_pca_components", 5)

    # If user didn’t whitelist explicit fields, build a default list
    if not fields:
        fields = [f"var_rat/{i}" for i in range(comps)] + [f"cum_var/{i}" for i in range(comps)] + [f"pc_mean/{i}" for i in range(comps)] + [f"pc_std/{i}" for i in range(comps)]
       
    # Cartesian product:  key/field
    return [f"{k}/{f}" for k in keys for f in fields]

@register_diagnostic(name = "layer_pca", field_fn = pca_field_fn) 
def layer_pca(model, val_loader, logger, epoch, cfg, meta, step, **kwargs):
    """
    Computes PCA over the latent vectors in the model output and logs explained variance ratios.
    Optionally logs a 2D PCA scatter plot.

    Config options (in cfg["diagnostics_config"]):
        - latent_pca_key (str): name of latent vector in model output dict (e.g., "latent")
        - latent_pca_components (int): number of PCA components (default: 5)
        - max_batches (int): max val batches to use
        - plot (bool): whether to generate a 2D PCA scatter plot (default: True)
S
    #TODO: Add per label details here. 
    """
    diag_cfg = cfg.get("diagnostics_config", {})
    layers = diag_cfg.get("layer_pca_layers", ["latent"])
    n_components = diag_cfg.get("layer_pca_components", 5)
    max_batches = diag_cfg.get("max_batches", 1)
    do_plot = diag_cfg.get("plot", True)
    
    model.eval()
    
    outputDict = {
    }
    
    for layer in layers: 
       
        latents, labels = compute_latent_all(model, val_loader, layer, max_batches)

        _, output_dict = run_pca_analysis(latents, labels, layer, logger, epoch, n_components, None, None,  do_plot, step)
       
        outputDict.update(output_dict)
        
    return outputDict

@register_diagnostic(name = "global_pca", field_fn = pca_field_fn) #TODO: Field function not right/used. 
def global_pca(model, val_loader, logger, epoch, cfg, meta, step, **kwargs):
    diag_cfg = cfg.get("diagnostics_config", {})
    layers = diag_cfg.get("layer_pca_layers", ["latent"])
    n_components = diag_cfg.get("layer_pca_components", 5)
    max_batches = diag_cfg.get("max_batches", 1)
    do_plot = diag_cfg.get("plot", True)
    num_latents = diag_cfg.get("num_latents", 20)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    model.eval()
    
    outputDict = {
    }
    
    for layer in layers:
        #TODO: fix this later - will need to do something more reproducible. 
        mean, components = meta.get(f"{layer}/mean", None), meta.get(f"{layer}/components", None)
        latent, labels = compute_latent_batch(model, val_loader, layer, seed, num_latents)
        _, output_dict = run_pca_analysis(latent, labels, f"{layer}", logger, epoch, n_components, (components, mean), None,  do_plot, step,)
        outputDict.update(output_dict)
        
    return outputDict

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .registry import register_diagnostic # Your decorator
import PIL.Image
from visualization import plot_pca_scree, plot_pca_component, plot_pca_2d_scatter, plot_pca_3d_scatter
import io
from utils.flatten import flatten
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
def layer_pca(model, val_loader, logger, epoch, cfg, meta):
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
    do_plot = diag_cfg.get("plot", True)

    model.eval()
    
    outputDict = {
    }
    
    for layer in layers: 
        print("layer: ", layer)
        output_dict = layer_pca_helper(model, val_loader, logger, epoch, layer, n_components, max_batches, do_plot, meta)
        outputDict.update(output_dict)
        
    return outputDict
def layer_pca_helper(model, val_loader,logger, epoch, layer, n_components, max_batches, do_plot, meta):
    """
    For now: One model call per different layer we want to check. Slow, but saves memory. 
    """
    outputDict = {}
    latents = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            inputs, targets = model.prepare_input(batch)
            targets = dict(flatten(targets))
            out = dict(flatten(model(**inputs)))
         
            z = out[layer].detach().cpu()
   
            latents.append(z)
            
            if targets.get("labels_y", None) is not None:
                labels.append(targets["labels_y"].detach().cpu())

    if not latents:
        return

    all_latents = torch.cat(latents, dim=0)
    all_labels = torch.cat(labels, dim=0) if labels else None
    pca = PCA(n_components=n_components)
    #the components are the coordinates right? 
    components = pca.fit_transform(all_latents.numpy())
    #dimensions? 
    explained_variance = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_variance)
    latent_dim_weights = pca.components_
    pca_mean = pca.mean_
    
    for i, var in enumerate(explained_variance):
        outputDict[f"{layer}/var_rat/{i}"] = var
    for i, cvar in enumerate(cum_var):
        outputDict[f"{layer}/cum_var/{i}"] = cvar
    for i in range(n_components):
        pc = components[:, i]
        outputDict[f"{layer}/pc_mean/{i}"] = np.mean(pc)
        outputDict[f"{layer}/pc_std/{i}"] = np.std(pc)
    #save the weights and the components. 
    
    logger.save_artifact(latent_dim_weights, f"{layer}/weights/weights_epoch_{epoch}")
    logger.save_artifact(components, f"{layer}/components/components_epoch_{epoch}")
    fig = plot_pca_scree(n_components, explained_variance, cum_var, layer)
    logger.save_plot(fig, f"{layer}/scree/scree_epoch_{epoch}")
    fig = plot_pca_component(n_components, latent_dim_weights)
    logger.save_plot(fig, f"{layer}/basis/basis_epoch_{epoch}")

    if do_plot and n_components >= 2:
        fig = plot_pca_2d_scatter(components, all_labels, layer)
        logger.save_plot(fig, f"{layer}/pca_scatter_2d/pca_scatter_2d_epoch_{epoch}")
    if do_plot and n_components>=3:
        fig = plot_pca_3d_scatter(components, all_labels, layer)
        logger.save_plot(fig, f"{layer}/pca_scatter_3d/pca_scatter_3d_epoch_{epoch}")
    return outputDict

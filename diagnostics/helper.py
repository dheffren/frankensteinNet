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
def run_pca_analysis(latents, labels, layer, logger, epoch, n_components, external_pca_basis, relative_basis, do_plot, step, meta = None):
    #%TODO: Fix global naming vs local naming. 
    outputDict = {}
    if external_pca_basis is not None and external_pca_basis != (None, None): 
        components, pca_mean = external_pca_basis
        projected = (latents - pca_mean) @  components.T
        logger.save_artifact(projected, f"{layer}/projectedExt/projected_epoch_{epoch}") #this one is special - do we not need to do this in the external case?
    else: 

        pca = PCA(n_components=n_components)
        #the components are the coordinates right? 
        projected = pca.fit_transform(latents.numpy())
        #dimensions? 
        explained_variance = pca.explained_variance_ratio_
        cum_var = np.cumsum(explained_variance)
        components = pca.components_
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
        #this is saving on the prerun. 
        if meta is not None:
            meta[f"{layer}/mean"] = pca_mean
            meta[f"{layer}/components"] = components
      
        #if that first run, mean-1. 
        logger.save_artifact(components, f"{layer}/weights/weights_epoch_{epoch}")
        logger.save_artifact(pca_mean, f"{layer}/mean/mean_epoch_{epoch}")
        #save projected and projected external in different spots. 

        logger.save_artifact(projected, f"{layer}/projected/projected_epoch_{epoch}") #this one is special - do we not need to do this in the external case?
        fig = plot_pca_scree(n_components, explained_variance, cum_var, layer)
        logger.save_plot(fig, f"{layer}/scree/scree_epoch_{epoch}.png", step)
        fig = plot_pca_component(n_components, components)
        logger.save_plot(fig, f"{layer}/basis/basis_epoch_{epoch}.png", step)
    #saving the projected part. 
    
    #track latent shift
    if relative_basis is not None:
        rel_proj = (latents - relative_basis["mean"]) @ relative_basis["components"].T
        shift = np.linalg.norm(projected - rel_proj, axis=1).mean()
        outputDict[f"{layer}/relative_shift"] = shift    
    
    if do_plot and n_components >= 2:
        fig = plot_pca_2d_scatter(projected, labels, layer)
        logger.save_plot(fig, f"{layer}/pca_scatter_2d/pca_scatter_2d_epoch_{epoch}.png", step)
    if do_plot and n_components>=3:
        fig = plot_pca_3d_scatter(projected, labels, layer)
        logger.save_plot(fig, f"{layer}/pca_scatter_3d/pca_scatter_3d_epoch_{epoch}.png", step)
    return projected, outputDict

        
def compute_latent_all(model, val_loader, layer, max_batches):
    """
    For now: One model call per different layer we want to check. Slow, but saves memory. 
    """
    
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
    return all_latents, all_labels
def compute_latent_batch(model, val_loader, layer, seed, num_samples = 12):
    labels = None
    with torch.no_grad():
        #supposedly this gives a fixed subset. 
        batch = get_fixed_batch(val_loader, seed, num_samples=num_samples)
        inputs, targets = model.prepare_input(batch)
        out = dict(flatten(model(**inputs)))
        latents = out[layer].detach().cpu()
        if targets.get("labels_y", None) is not None:
            labels= targets["labels_y"].detach().cpu()
    return latents, labels

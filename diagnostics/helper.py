import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .registry import register_diagnostic # Your decorator
import PIL.Image
from diagnostics.visualization import plot_pca_scree, plot_pca_component, plot_pca_2d_scatter, plot_pca_3d_scatter
import io
from utils.flatten import flatten
from utils.fixedBatch import get_fixed_batch
def run_pca_analysis(latents, labels, layer, n_components, external_pca_basis, relative_basis, do_plot, meta = None):
    #%TODO: Fix global naming vs local naming. 
    outputDict = {}
    artifactList = []
    calledHere = False
    var=  isinstance(external_pca_basis, tuple)
  
    if var:
        print(isinstance(external_pca_basis[0], np.ndarray))
    if (isinstance(external_pca_basis, tuple) and isinstance(external_pca_basis[0], np.ndarray)):
    #if external_pca_basis is not None and external_pca_basis != (None, None): 
        components, pca_mean = external_pca_basis
        projected = (latents - pca_mean) @  components.T
        artifactList.append({"kind":"bytes", "key": f"{layer}/projectedExt", "data":projected, "split": "val"})
        calledHere = True
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
        print("got to meta")
      
        if meta is not None:
            print("in meta")
            meta[f"{layer}/mean"] = pca_mean
            meta[f"{layer}/components"] = components
      
        #if that first run, mean-1. 
        artifactList.append({"kind":"bytes", "key": f"{layer}/weights", "data":components, "split": "val"})
        artifactList.append({"kind":"bytes", "key": f"{layer}/mean", "data":pca_mean, "split": "val"})
        artifactList.append({"kind":"bytes", "key": f"{layer}/projected", "data":projected, "split": "val"})
        
        fig = plot_pca_scree(n_components, explained_variance, cum_var, layer)
        artifactList.append({"kind":"figure", "key": f"{layer}/scree", "fig":fig, "split": "val"})
        
        fig = plot_pca_component(n_components, components)
        artifactList.append({"kind":"figure", "key": f"{layer}/basis", "fig":fig, "split": "val"})
        
    #saving the projected part. 
    
    #track latent shift
    if relative_basis is not None:
        rel_proj = (latents - relative_basis["mean"]) @ relative_basis["components"].T
        shift = np.linalg.norm(projected - rel_proj, axis=1).mean()
        outputDict[f"{layer}/relative_shift"] = shift    
    
    if do_plot and n_components >= 2:
        fig = plot_pca_2d_scatter(projected, labels, layer)
        artifactList.append({"kind":"figure", "key": f"{layer}/pca_scatter_2d", "fig":fig, "split": "val"})
        
    if do_plot and n_components>=3:
        fig = plot_pca_3d_scatter(projected, labels, layer)
        artifactList.append({"kind":"figure", "key": f"{layer}/pca_scatter_3d", "fig":fig, "split": "val"})
    return projected, outputDict, artifactList

        
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

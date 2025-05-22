import torch
from utils.flatten import flatten
from .registry import register_diagnostic
from utils.fixedBatch import get_fixed_batch
@register_diagnostic() 
def weight_perturb(model, val_loader, logger, epoch, cfg, meta):
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
    direction_types = diag_cfg.get("direction_types", ['random', 'gradient'])
    num_dirs = diag_cfg.get("num_dirs", 4)
    #heatmap_dirs = 
    epsilons = diag_cfg.get("epsilons", [1e-4, 1e-3, 1e-2])
    num_latents = diag_cfg.get("num_latents", 20)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    model.eval()
    eps = .01    
    outputDict = {
    }
    batch = get_fixed_batch(val_loader, seed, num_samples= num_latents)
    inputsbase, targetbase = model.prepare_input(batch)
    out_base = dict(flatten(model(**inputs)))
    loss_dict_base = model.compute_loss_helper(out_base, target, epoch)
    for direction_type in direction_types: 
        dir = get_direction(model, batch, direction_type, epoch)
        for eps in epsilons:
            
            saved_weights = perturb_weights(model, dir, eps)
            inputs, target = model.prepare_input(batch)
            out = model(**inputs)
            outd = dict(flatten(out))
            loss_dict = model.compute_loss_helper(out, target, epoch)
            #just keep track of OVERALL loss right now. 
            outputDict[f"{direction_type}_{eps}"] = loss_dict["loss"]

            #do latent shift here - mightneed pca. 
            #TODO: Visualize latent evolution. 
            # Visualize Recon. 
            #what's a good way to do this? Maybe each row is original image - then original output, then 
            #TODO: Add reconstructions
            #logger.log_plot()

            restore_weights(model, saved_weights)
    #heatmap = compute_loss_surface_heatmap(model, batch, directions[])
    return outputDict
def get_directions(model, batch, direction_list):
    #need a method to get a list of directions so i can do other stuff. 
    return
def get_direction(model, batch, direction, epoch):
    if direction == 'random': 
        #CHECK THE NORMS HERE. 
        return [torch.randn_like(p) for p in model.parameters() if p.requires_grad]
    elif direction == "gradient":
        inputs, target = model.prepare_input(batch)
        out_base = dict(flatten(model(**inputs)))
        loss_dict_base = model.compute_loss_helper(out_base, target, epoch)
        grads = torch.autograd.grad(loss_dict_base["loss"], [p for p in model.parameters() if p.requires_grad], create_graph=False)
        return grads
    else: 
        raise NotImplementedError


def restore_weights(model, saved_weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), saved_weights):
            if p.requires_grad:
                p.copy_(w)    
def perturb_weights(model, dir,eps):
    saved_weights = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    # Apply perturbation
    with torch.no_grad():
        for p, d in zip(model.parameters(), dir):
            if p.requires_grad:
                p.add_(eps * d)
        return saved_weights
def loss_surface_heatmap():
    return

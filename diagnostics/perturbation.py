import torch
from utils.flatten import flatten
from .registry import register_diagnostic
from utils.fixedBatch import get_fixed_batch
import numpy as np
import matplotlib.pyplot as plt
@register_diagnostic()
def loss_surface_heatmap(model, val_loader, logger, epoch, cfg, meta,step, **kwargs):
    diag_cfg = cfg.get("diagnostics_config", {})
    layers = diag_cfg.get("layer_pca_layers", ["latent"])
    n_components = diag_cfg.get("layer_pca_components", 5)
    max_batches = diag_cfg.get("max_batches", 1)
    save_latents = diag_cfg.get("save_latents", False)
    direction_types = diag_cfg.get("direction_types", ['random', 'gradient'])
    num_dirs = diag_cfg.get("num_dirs", 4)
    #heatmap_dirs = 

    num_latents = diag_cfg.get("num_latents", 20)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    model.eval()
    eps = .0001   
    outputDict = {
    }
    batch = get_fixed_batch(val_loader, seed, num_samples= num_latents)
    #get the inputs and target for the default batch. 
    

    tau = .001
    #loss_dict_base = model.compute_loss_helper(out, targetbase, epoch) #epoch doesn't make sense as a post training thing. 
    print('starting weight perturb')
   
    #TODO: Think about metric here when doing GS or QR - could weight different params by different amounts. 
    # COuld also use two lanczos directions ie top 2 hessian eigenvectors. 
    raw1 = make_random_direction(model)
    raw2 = make_random_direction(model)
    v1, v2 = qr_two(raw1, raw2) #two orthonormal vectors. '
    n = 25
    half = n//2
    lossGrid = torch.zeros(n, n)
  
    epsilonhalf = (torch.arange(0, half+1)/half)*eps
    epsilons = torch.cat([-1*(torch.flip(epsilonhalf[1:], (0,))), epsilonhalf])
   
    for i in range(n):
        for j in range(n):
   
            alpha = ((i - half)/half)*eps
            beta = ((j - half)/half)*eps
            #print("Alpha: ", alpha)
            #print("Beta: ", beta)
            v = alpha*v1 + beta*v2
            #might work like this. 
            #TODO: Check memory before and after is alright. 
            saved_weights = perturb_weights_alt(model, v)
     
            #inputs, target = model.prepare_input(batch)
            loss = model.compute_loss(batch, epoch)["loss"].detach() #NEED DETACH TO STOP TRACKING GRADIENTS. 
            print('loss: ', loss)# not computing loss to the right level of precision
      
            #loss = model.compute_loss_helper(out, target, epoch)["loss"]
            print("restore weights")
            restore_weights(model, saved_weights)
       
            lossGrid[i, j] = loss

    print("loss Grid: ", lossGrid)
    # GRAPH THE HEATMAP
    fig = graph_heatmap(lossGrid, epsilons, epsilons)
    print("Step: ", step)
    logger.save_plot(fig, f"loss_landscape_epoch_{epoch}.png", step)
    #Draw reconstructions in same grid format. - can't do with dual structure. 
def check_mem():
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    # Convert to GB for readability
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    print(f"Free GPU memory: {free_gb:.2f} GB")
    print(f"Total GPU memory: {total_gb:.2f} GB")

def graph_heatmap(lossGrid, alphas, betas, title = "loss surface", cmap = "viridis"):
    Z = lossGrid.detach().cpu().numpy()
    A, B = np.meshgrid(alphas, betas)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Filled contour or imshow — here we use contourf for smoother look
    c = ax.contourf(A, B, Z, levels=50, cmap=cmap)
    fig.colorbar(c, ax=ax, label="Loss")

    # Mark the origin (unperturbed point)
    ax.plot(0, 0, "ro", markersize=6, label="Origin")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)

    ax.set_xlabel(r"$\alpha$ (dir 1 scale)")
    ax.set_ylabel(r"$\beta$ (dir 2 scale)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig

def make_random_direction(model):
    #TODO: Consider seed here? 
    #TODO: This normalizes each direction separately before normalizing the whole thing. 
    #v = torch.cat([torch.randn_like(p).reshape(-1) for p in model.parameters() if p.requires_grad])
    listDirs = []
    for p in model.parameters():
        if p.requires_grad:
            
            val = torch.randn_like(p).reshape(-1)
            listDirs.append(val/val.norm() + 1e-12)
    longVec = torch.cat(listDirs)
    v = longVec/ longVec.norm() # Normalize
    
    return v
def qr_two(u1, u2):
    M = torch.stack([u1, u2], dim=1)             # [D, 2]
    Q, _ = torch.linalg.qr(M, mode='reduced')     # Q: [D,2] with orthonormal cols
    return Q[:,0].contiguous(), Q[:,1].contiguous()
def input_perturb(model, val_loader, logger, epoch, cfg, meta, **kwargs):
    """
    Goal: Estimate lipschitz constant of encoder w.r.t inputs. This tells us how "smooth" the function is. 
    However, this is teh same as computing the operator norm of the jacobian. 

    """
@register_diagnostic() 
def weight_perturb(model, val_loader, logger, epoch, cfg, meta, **kwargs):
    """
    Goal: Estimate lipschitz constant of the encoder w.r.t the weights. This tells us something about how "smooth" 
    the network outputs vary with respect to the weights.  
    """
    diag_cfg = cfg.get("diagnostics_config", {})
    layers = diag_cfg.get("layer_pca_layers", ["latent"])
    n_components = diag_cfg.get("layer_pca_components", 5)
    max_batches = diag_cfg.get("max_batches", 1)
    save_latents = diag_cfg.get("save_latents", False)
    direction_types = diag_cfg.get("direction_types", ['random', 'gradient'])
    num_dirs = diag_cfg.get("num_dirs", 4)
    #heatmap_dirs = 
    epsilons = diag_cfg.get("epsilons", [1e-4, 5e-4, 1e-3])
    num_latents = diag_cfg.get("num_latents", 20)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    model.eval()
    eps = .01    
    outputDict = {
    }
    #get a fixed batch
    batch = get_fixed_batch(val_loader, seed, num_samples= num_latents)
    #get the inputs and target for the default batch. 
    inputsbase, targetbase = model.prepare_input(batch)
    out = model(**inputsbase)
    tau = .001
    out_base = dict(flatten(out))

    loss_dict_base = model.compute_loss_helper(out, targetbase, epoch)
    print('starting weight perturb')
    for direction_type in direction_types: 
     
        dir = get_direction(model, batch, direction_type, epoch)
      
        for eps in epsilons:
     
            saved_weights = perturb_weights(model, dir, eps)
            inputs, target = model.prepare_input(batch)
            out = model(**inputs)
            outd = dict(flatten(out))
            
            loss_dict = model.compute_loss_helper(out, target, epoch)
            print("loss: ", loss_dict["loss"])
            outputDict[f"rel_loss_{direction_type}_{eps}"] = (loss_dict["loss"] - loss_dict_base["loss"])/(loss_dict_base["loss"] + tau)
            outputDict[f"curvature_{direction_type}_{eps}"] = (2*outputDict[f"rel_loss_{direction_type}_{eps}"])/(eps**2) # Curvature proxy. 
            

            # Visualize Recon. 

           
            restore_weights(model, saved_weights)
            print("finished restore weights. ")
            #TODO: Check equality or original and reconstruction. 
    #heatmap = compute_loss_surface_heatmap(model, batch, directions[])
    return outputDict
def lipschitz(out, out_base, epsilon, dir, tau):
    """
    out is a specific layer or latent or output of the model. Not in dict form anymore. Call this on EACH THING you want to look at. 

    RN math only makes sense if out is batchSize x d. 
    """
    latent_shift = torch.norm(out - out_base, dim = -1)/torch.norm(out_base + tau, dim = -1)
    #avg shift in latent relative to size of latent. 
    mean_latent_shift = latent_shift.mean()
    std_latent_shift = latent_shift.std()
    iqr_latent_shift = latent_shift.quartile(.75) - latent_shift.quartile(.25)
    lipschitzConst = torch.norm(out - out_base, dim = -1)/epsilon
    mean_lipschitz = lipschitzConst.mean()
    std_lipschitz = lipschitzConst.std()
    iqr_lipschitz = lipschitzConst.quartile(.75) - lipschitzConst.quartile(.25)

def get_direction(model, batch, direction, epoch):
    if direction == 'random': 
        #CHECK THE NORMS HERE. 
        return [torch.randn_like(p) for p in model.parameters() if p.requires_grad]
    elif direction == "gradient":
        inputs, target = model.prepare_input(batch)
        out_base = model(**inputs)
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
def perturb_weights_alt(model, v):
    saved_weights =  [p.detach().clone() for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        index = 0
        for p in model.parameters():
            if p.requires_grad:
                n = p.numel()
                p.add_(v[index:index + n].view_as(p)) #need view_as because p is;t flat. 
                index = index + n
        return saved_weights
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

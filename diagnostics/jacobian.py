import torch
from .suffix_fn_registry import get_suffix_fn
from .registry import register_diagnostic

@register_diagnostic()
def jacobian_norm_diag(model, dataloader, logger, epoch, config):
    #TODO: Fix this so the things it's "calling" are in this file insetad of in that suffix file. 
    cfg = config.get("diagnostics_config", {})
    keys = cfg.get("jacobian_norm_diag_keys", [])
    suffixes = cfg.get("jacobian_norm_diag_suffixes", ["fro"])
    max_batches = cfg.get("max_batches", 1)

    if isinstance(keys, str): keys = [keys]
    if isinstance(suffixes, str): suffixes = [suffixes]

    model.eval()

    outputDict = {}
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        inputs, targets = model.prepare_input(batch)
        #not sure about this. Do i need to detach? 
        inputs["x"] = inputs["x"].requires_grad_(True)
        out = model(**inputs)
        for key in keys:
            for suffix in suffixes:
                #PRoblem is with submodules - the input for the decoder is DIFFERENT than the input for the encoder. 
                fn = get_suffix_fn(suffix)
                #submodule = getattr(model, key, None)
                #if submodule is None:
                    #print(f"[JacobianDiag] Warning: No module '{key}' in model.")
                    #continue
                try:
                    #val = #fn(model, x, key)
                    val = fn(inputs["x"], out[key])
                    outputDict[f"{key}/{suffix}"] = val
                    
                except Exception as e:
                    print(f"[JacobianDiag] Failed for {key}/{suffix}: {e}")
    return outputDict#]
def compute_jacobian_fro_norm(x, y):
    #problem was that this was under nograd.
    for i in range(y.shape[1]):
        #element 0 of tensors does not require grad and does not have a grad_fn
        grad = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True, create_graph=False)[0]
        J.append(grad.view(grad.shape[0], -1))  # (B, D)
    J = torch.stack(J, dim=1)  # (B, output_dim, input_dim)
    norms = torch.norm(J, dim=(1, 2), p='fro')  # (B,)
    return norms.mean().item()

def compute_jacobian_spectral_norm(x, y, num_iters = 5, eps=1e-8):
    """
    Estimates the spectral norm (largest singular value) of the Jacobian dy/dx
    using power iteration, without modifying .grad buffers.

    Args:
        model: the neural network
        x (Tensor): input tensor of shape (B, ...)
        key (str): key to extract relevant output if model returns dict
        num_iters (int): number of power iteration steps
        eps (float): small constant for numerical stability

    Returns:
        float: estimated spectral norm (average over batch)

    CLAIM: Can't use autograd to compute spectral norm because of the way grad_output works. 
    Usefunctorch instead. 

    Here, we use a "single step estimate" instead. 
    This approximates the spectral norm with a lower bound. 

    Could do SVD if the latent dim was small. 
    """
    #flatten to either recon size or latent size. 
    y = y.view(x.size(0), -1)  # shape (B, D_out)
    B, D_out = y.shape

    # Initialize random output-space vector (for Jacobian-vector product)
    v = torch.randn(B, D_out, device=x.device)
    v = v / (v.norm(dim=1, keepdim=True) + eps)

   
    # Compute J^T v (shape: same as x)
    JTv = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=v,
        retain_graph=True, create_graph=False
    )[0]
    return  JTv.view(B, -1).norm(dim=1).mean().item()
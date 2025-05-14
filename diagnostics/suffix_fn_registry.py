import torch
import torch.nn.functional as F

# Scalar computation functions
def compute_mse(x, y):
    return F.mse_loss(x, y).item()

def compute_mae(x, y):
    return F.l1_loss(x, y).item()
def compute_jacobian_fro_norm(x, y):
    """
    Assumptions on inputs:
    y is a function of x. 
    y seems to be batchSize x d. 
    What if we wanted a different shape for y? 
    x can be any shape batchSize x dim1 x dim2 x.... dim k 
    J we stack 
    
    """
    J = []
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
# Registry of suffixes and how to compute them
SUFFIX_FN_REGISTRY = {
    "mean": lambda t: t.mean().item(),
    "std": lambda t: t.std().item(),
    "min": lambda t: t.min().item(),
    "max": lambda t: t.max().item(),
    "var": lambda t: t.var().item(),
    "fro": lambda t: torch.norm(t, p='fro').item() if t.ndim >= 2 else t.norm().item(),
    "entropy": lambda t: -(t * (t + 1e-8).log()).sum().item() if (t > 0).all() else float("nan"),
    "mse": compute_mse,
    "mae": compute_mae,
    "spectral":compute_jacobian_spectral_norm, 
    "jacfro":compute_jacobian_fro_norm
}

# Which suffixes require (x, y) inputs (e.g. recon + target)
SUFFIX_PAIRWISE = {"mse", "mae"}

def get_available_suffixes():
    return list(SUFFIX_FN_REGISTRY)

def is_pairwise_suffix(suffix):
    return suffix in SUFFIX_PAIRWISE

def get_suffix_fn(suffix):
    if suffix not in SUFFIX_FN_REGISTRY:
        raise ValueError(f"Unknown suffix: '{suffix}'. Available: {list(SUFFIX_FN_REGISTRY)}")
    return SUFFIX_FN_REGISTRY[suffix]

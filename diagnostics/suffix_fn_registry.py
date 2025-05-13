import torch
import torch.nn.functional as F

# Scalar computation functions
def compute_mse(x, y):
    return F.mse_loss(x, y).item()

def compute_mae(x, y):
    return F.l1_loss(x, y).item()
#TODO: Fix the model output - it wants me to put in the encoder or decoder, but the problem with that is it doesn't make sense with what i'm doing. 
def compute_jacobian_fro_norm(model, x):
    x = x.requires_grad_(True)
    y = model(x)
    if isinstance(y, dict): y = y["latent"]  # or whatever latent
    J = []
    for i in range(y.shape[1]):
        grad = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True, create_graph=False)[0]
        J.append(grad.view(grad.shape[0], -1))  # (B, D)
    J = torch.stack(J, dim=1)  # (B, output_dim, input_dim)
    norms = torch.norm(J, dim=(1, 2), p='fro')  # (B,)
    return norms.mean().item()

def compute_jacobian_spectral_norm(model, x):
    # Approximate via power iteration (very rough)
    x = x.requires_grad_(True)
    y = model(x)
    if isinstance(y, dict): y = y["latent"]
    vec = torch.randn_like(x)
    vec = vec / vec.norm()
    for _ in range(5):
        vec = torch.autograd.grad(y.sum(), x, grad_outputs=vec, retain_graph=True)[0]
        vec = vec / vec.norm()
    return vec.norm().item()
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

import torch
import torch.nn.functional as F

# Scalar computation functions
def compute_mse(x, y):
    return F.mse_loss(x, y).item()

def compute_mae(x, y):
    return F.l1_loss(x, y).item()

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
# losses.py
"""Modular loss‑function factory & building blocks.

Usage
-----
>>> loss_fn = make_loss_fn(cfg["loss"])   # cfg comes from YAML
>>> loss_dict = loss_fn(model.forward, batch)  # inside model.compute_loss()

Each `make_*_loss` factory returns a callable that expects two arguments:
    forward_fn  – usually `model.forward` or a wrapper.
    batch       – whatever the dataloader yields (tensor / (x,y) / dict).
It returns a dictionary **must** contain a key "total" (torch scalar) and
may additionally include any diagnostic scalars you wish to log.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 🎯  Generic loss helpers / primitives
# -----------------------------------------------------------------------------


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    #deal with shape mismatches here? TODO: Check to make sure this is doing the right thing. 
    """Mean‑squared error with mean reduction."""
    return F.mse_loss(x_hat, x, reduction="mean")


def bce_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Binary‑cross‑entropy (pixel‑wise). Expects x to be in [0,1]."""
    return F.binary_cross_entropy(x_hat, x, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Analytic KL(N(mu, σ²) || N(0,1)) per batch element then mean."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# -----------------------------------------------------------------------------
# 🎯  Loss factories
# -----------------------------------------------------------------------------


def make_vae_loss(kl_weight: float = 1.0,
                  recon_type: str = "mse",
                  **extra: Any) -> Callable[[Callable, Any], Dict[str, torch.Tensor]]:
    """Factory for classic VAE loss.
    Extra keys are ignored with a warning (helps catch typos).
    """
    if extra:
        warnings.warn(f"[make_vae_loss] Unused keys in loss config: {list(extra.keys())}")

    if recon_type not in {"mse", "bce"}:
        raise ValueError(f"Unsupported recon_type '{recon_type}'. Use 'mse' or 'bce'.")

    recon_fn = mse_loss if recon_type == "mse" else bce_loss

    def _loss_fn(batch, forward_fn: Callable, ) -> Dict[str, torch.Tensor]:
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        out = forward_fn(x)
        recon = out["recon"]
        mu, logvar = out["mu"], out["logvar"]

        recon_loss = recon_fn(x, recon)
        kl = kl_divergence(mu, logvar)
        total = recon_loss + kl_weight * kl

        return {"loss": total, "recon": recon_loss.detach(), "kl": kl.detach()}

    return _loss_fn


def make_ae_loss(recon_type: str = "mse", **extra) -> Callable[[Callable, Any], Dict[str, torch.Tensor]]:
    if extra:
        warnings.warn(f"[make_ae_loss] Unused keys in loss config: {list(extra.keys())}")
    recon_fn = mse_loss if recon_type == "mse" else bce_loss

    def _loss_fn( batch, forward_fn: Callable,):
        #checks if tuple or not. 
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        outputDict = forward_fn(x)
        recon = outputDict["recon"]
        latent = outputDict["latent"]
        total = recon_fn(x, recon)
        #ADD EXTRA THINGS REGARDING LATENT HERE. 

        #make sure these names are consistent. 
        return {"loss": total}

    return _loss_fn

def backup_loss( batch, forward):
        #TODO: Fix this structure. 
        x, y = batch
        reconstruction, latent = forward(x)
      
        l2loss = torch.mean(torch.sum((reconstruction - x)**2, axis=(-1, -2)))
        loss_dict = {"loss": l2loss}
        return loss_dict

def make_contrastive_loss(temperature: float = 0.07, **extra):
    if extra:
        warnings.warn(f"[make_contrastive_loss] Unused keys in loss config: {list(extra.keys())}")

    def _loss_fn( batch, forward_fn: Callable, ):
        # Expect forward_fn to return L2‑normalised embeddings of shape [B, D]
        z = forward_fn(batch)
        sim = torch.mm(z, z.t()) / temperature  # [B,B]
        labels = torch.arange(z.size(0), device=z.device)
        loss = F.cross_entropy(sim, labels)
        return {"loss": loss}

    return _loss_fn


# -----------------------------------------------------------------------------
# 🎯  Dispatcher – choose loss factory by config
# -----------------------------------------------------------------------------

LOSS_FACTORY = {
    "vae": make_vae_loss,
    "ae": make_ae_loss,
    "contrastive": make_contrastive_loss,
}


def make_loss_fn(loss_cfg: Dict[str, Any]) -> Callable[[Callable, Any], Dict[str, torch.Tensor]]:
    """Create a loss function based on YAML/JSON 'loss' block.

    Parameters
    ----------
    loss_cfg : dict
        Must contain at least a key ``type``. Additional keys are passed to the
        selected factory. Unknown keys are ignored by the factory unless it
        chooses to warn on them.
    """
    if "type" not in loss_cfg:
        raise KeyError("loss_cfg must contain a 'type' field (e.g., 'vae').")
    #loss type: 
    loss_type = loss_cfg["type"].lower()
    #get the loss type
    factory = LOSS_FACTORY.get(loss_type)
    if factory is None:
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {list(LOSS_FACTORY)}")
    #get params for 
    kwargs = {k: v for k, v in loss_cfg.items() if k != "type"}
    return factory(**kwargs)

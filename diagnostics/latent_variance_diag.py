import torch
from .registry import register_diagnostic
@register_diagnostic()
def latent_variance_diag(model, val_loader, logger, epoch, cfg):
    model.eval()
    zs = []
    with torch.no_grad():
        for x, *_ in val_loader:
            out = model(x.to(logger.device))
            if "z" not in out: return
            zs.append(out["z"].cpu())
    z = torch.cat(zs)
    var = z.var(0).mean().item()
    logger.log_scalar("diag/latent_variance", var, epoch)
    from registry import DIAGNOSTIC_FIELD_REGISTRY
    DIAGNOSTIC_FIELD_REGISTRY["latent_variance_diag"] = [
        f"diag/latent_variance"
    ]
from visualization import make_reconstruction_plot
import torch

# fn(self.model, self.val_loader, self.logger, epoch, self.config)
from .registry import register_diagnostic

@register_diagnostic()
def log_reconstruction_plot(model, dataloader, logger, epoch, config):
    #reconstruction plot diagnostic. 
    model.eval()
    #maybe see if this should ahve a default or not? 
    #fixed this so it works with the new registry system. 
    diag_cfg = config.get("diagnostics_config", {})
    num_images = diag_cfg.get("num_recon_samples", 8)
    # Grab a batch of data
    #Note: Designed so this is agnostic to the type of data passed in. 
    #always takes first batch of validation (and same samples)
    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs, target = model.prepare_input(batch)
        out = model(**inputs)

    x_batch = target["x"]
    x_recon = out["recon"].cpu()
    fig = make_reconstruction_plot(x_batch, x_recon, epoch, num_images)
    logger.save_plot(fig, f"recon_epoch_{epoch}.png")

def unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    elif isinstance(batch, dict):
        return batch.get("input", next(iter(batch.values())))
    elif isinstance(batch, torch.Tensor):
        return batch
    else:
        raise TypeError(f"Unknown batch format: {type(batch)}")
def get_reconstruction_output(model, x):
    #Instead of doing this this way, can make a base class standardizing .reconstruct for all models. 
    out = model(x)
    if isinstance(out, torch.Tensor):
        #should I check if they're the same shape somewhere? 
        return out
    # 2. Tuple: assume recon is first
    elif isinstance(out, (tuple, list)):
        return out[0]

    # 3. Dict: try common keys
    elif isinstance(out, dict):
        for key in ["recon", "reconstruction", "x_hat", "output"]:
            if key in out:
                return out[key]
        raise ValueError("Model dict output missing expected reconstruction key")

    else:
        raise TypeError(f"Unrecognized model output type: {type(out)}")
    return 
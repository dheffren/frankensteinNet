from visualization import make_reconstruction_plot, make_dual_reconstsruction_plot
import torch

# fn(self.model, self.val_loader, self.logger, epoch, self.config)
from .registry import register_diagnostic
from utils.flatten import flatten
@register_diagnostic()
def log_reconstruction_plot(model, dataloader, logger, epoch, config, meta):
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
    fig = handle_reconstructions(target, out,  epoch, num_images, meta)
    logger.save_plot(fig, f"recon_epoch_{epoch}.png")
def handle_reconstructions(target, out, epoch, num_images, meta):
    x = target["recon_target"]
    recon = out["recon"]
    #DO I need detach
    if len(x.keys()) == 1 and len(recon.keys()) == 1:
        return make_reconstruction_plot(x["x"], recon["x"].cpu(), epoch, num_images, meta)
    elif len(x.keys()) == 2 and len(recon.keys()) == 2:
        return make_dual_reconstsruction_plot(x["x1"], recon["x1"].cpu(), x["x2"], recon["x2"].cpu(), epoch, num_images, meta)
    else: 
        raise ValueError("x and recon have the wrong format")
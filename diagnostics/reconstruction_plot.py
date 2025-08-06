from visualization import make_reconstruction_plot, make_dual_reconstsruction_plot
import torch

# fn(self.model, self.val_loader, self.logger, epoch, self.config)
from .registry import register_diagnostic
from utils.flatten import flatten
from utils.fixedBatch import get_fixed_batch
@register_diagnostic()
def log_reconstruction_plot(model, val_loader, logger, epoch, cfg, meta, step, **kwargs):
    #reconstruction plot diagnostic. 
    model.eval()
    #maybe see if this should ahve a default or not? 
    #fixed this so it works with the new registry system. 
    diag_cfg = cfg.get("diagnostics_config", {})
    num_images = diag_cfg.get("num_recon_samples", 8)
    seed = diag_cfg.get("fixed_batch_seed", 32)
    # Grab a batch of data
    #Note: Designed so this is agnostic to the type of data passed in. 
    #always takes first batch of validation (and same samples)
    with torch.no_grad():
        batch = get_fixed_batch(val_loader, seed, num_samples= num_images)
        inputs, target = model.prepare_input(batch)
        out = model(**inputs)
    fig = handle_reconstructions(target, out,  epoch, num_images, meta)
    logger.save_plot(fig, f"recon_epoch_{epoch}.png", step)
def handle_reconstructions(target, out, epoch, num_images, meta):
    x = target["recon_target"]
    recon = out["recon"]

    normalizer = meta.get("normalizer", None)
    if normalizer: 
        #will recognize either the x1 or the x2 or the x keys and renormalize accordingly. 
        x = normalizer.denormalize(x)
        #bring to cpu. 
        recon = {k:v.cpu() for k,v in recon.items()}
        #same for recon: will have keys x1, x2 or x or whatever the corresponding input was. 
        recon = normalizer.denormalize(recon)
    #DO I need detach
    if len(x.keys()) == 1 and len(recon.keys()) == 1:
        return make_reconstruction_plot(x["x"], recon["x"].cpu(), epoch, num_images, meta)
    elif len(x.keys()) == 2 and len(recon.keys()) == 2:
        
        return make_dual_reconstsruction_plot(x["x1"], recon["x1"].cpu(), x["x2"], recon["x2"].cpu(), epoch, num_images, meta)
    else: 
        raise ValueError("x and recon have the wrong format")
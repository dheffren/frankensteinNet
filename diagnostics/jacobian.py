import torch
from .suffix_fn_registry import get_suffix_fn
from .registry import register_diagnostic

@register_diagnostic()
def jacobian_norm_diag(model, dataloader, logger, epoch, config):
    cfg = config.get("diagnostics_config", {})
    keys = cfg.get("jacobian_norm_diag_keys", [])
    suffixes = cfg.get("jacobian_norm_diag_suffixes", ["fro"])
    max_batches = cfg.get("max_batches", 1)

    if isinstance(keys, str): keys = [keys]
    if isinstance(suffixes, str): suffixes = [suffixes]

    model.eval()

    outputDict = {}
    for i, (x, *_) in enumerate(dataloader):
        if i >= max_batches:
            break
        x = x.to(config["device"])
        for key in keys:
            for suffix in suffixes:
                #PRoblem is with submodules - the input for the decoder is DIFFERENT than the input for the encoder. 
                fn = get_suffix_fn(suffix)
                #submodule = getattr(model, key, None)
                #if submodule is None:
                    #print(f"[JacobianDiag] Warning: No module '{key}' in model.")
                    #continue
                try:
                    val = fn(model, x, key)
                    outputDict[f"{key}/{suffix}"] = val
                    
                except Exception as e:
                    print(f"[JacobianDiag] Failed for {key}/{suffix}: {e}")
    return outputDict

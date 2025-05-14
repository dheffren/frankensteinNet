import torch
from .suffix_fn_registry import get_suffix_fn, is_pairwise_suffix
from .registry import register_diagnostic

#TODO: More gneeral? 
suffix_fns = {
    "mean": lambda t: t.mean().item(),
    "std":  lambda t: t.std().item(),
    "min":  lambda t: t.min().item(),
    "max":  lambda t: t.max().item(),
}

@register_diagnostic()
def tensor_stats_diag(model, val_loader, logger, epoch, cfg):
    """
    Logs mean, std, min, max of a specified tensor in model output.
    Requires `cfg["diagnostics"]["tensor_stats_key"]` to be set.
    """
    diag_cfg = cfg.get("diagnostics_config", {})
    keys = diag_cfg.get("tensor_stats_diag_keys", [])
    suffixes = diag_cfg.get("tensor_stats_diag_suffixes", list(suffix_fns))
    max_batches = diag_cfg.get("max_batches", 1) # designed to limit the number of batches called. 
    
    if isinstance(keys, str):
        keys = [keys]
    model.eval()
    data_by_key = {k: [] for k in keys}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            inputs, target = model.prepare_input(batch)
            out = model(**inputs)
            for k in keys:
                if k in out:
                    data_by_key[k].append(out[k].detach().cpu())
    outputDict = {}
    for k, tensors in data_by_key.items():
       if not tensors:# what is this for again? 
           continue
       t = torch.cat(tensors)
       for suffix in suffixes:
            fn = get_suffix_fn(suffix)
            if is_pairwise_suffix(suffix):
                #May want to passin x here. 
                val = fn(t,t)
            else:
                val = fn(t)
            outputDict[f"{k}/{suffix}"] = val
    return outputDict
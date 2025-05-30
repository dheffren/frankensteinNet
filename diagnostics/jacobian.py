import torch
from .suffix_fn_registry import get_suffix_fn
from .registry import register_diagnostic
from utils.flatten import flatten, safe_keyname
from utils.fixedBatch import get_fixed_batch
from .helper import *
def jac_field_fn(cfg:dict):
    pairs = cfg.get("jacobian_norm_diag_pairs", [])
    suffixes = cfg.get("jacobian_norm_diag_suffixes", ["fro"])
    print(pairs)
    return [f"{p['of']}_wrt_{p['wrt']}/{suff}" for p in pairs for suff in suffixes]
#TODO: Automatic field naming isn't working here - using dynamic. It works for now, but not intended. I think maybe ALL field naming being dynamic would be better. 
@register_diagnostic(name = "jacobian_norm_diag", field_fn = jac_field_fn)
def jacobian_norm_diag(model, val_loader, logger, epoch, cfg, meta, **kwargs):
    #TODO: Fix this so the things it's "calling" are in this file insetad of in that suffix file. 
    """
    Note: 
    want this method for each key and suffix - (keys might be the latent spaces of any of them). But that means they may have different
    Inputs for each. 
    TODO: Need to require gradients so this will work - more problems. 
    """
    dcfg = cfg.get("diagnostics_config", {})

    pairs = dcfg.get("jacobian_norm_diag_pairs", [])
    suffixes = dcfg.get("jacobian_norm_diag_suffixes", ["fro"])
    max_batches = dcfg.get("max_batches", 1)
    seed = dcfg.get("seed", 32)
    num_latents = dcfg.get("num_latents", 20)
    model.eval()

    outputDict = {}
    
    batch = get_fixed_batch(val_loader, seed, num_samples= num_latents)
    inputs, targets = model.prepare_input(batch, requires_grad = True)
    targets = dict(flatten(targets))
    #how do I do this IN the model?
    out = dict(flatten(model(**inputs)))
    #print(pairs)
    for pair in pairs: 
        #print("pair: ", pair)
        for suffix in suffixes:
            #PRoblem is with submodules - the input for the decoder is DIFFERENT than the input for the encoder. 
            fn = get_suffix_fn(suffix)
            
            try:
                
                #how can I avoid directly inputting x here? 
                #need to also deal with the problems of this should be fieldnames
                x = pair["of"]
                y = pair["wrt"]
                #print(x)
                #print(y)
                if x in inputs.keys():
                    xh = inputs[x]
                elif x in targets.keys():
                    xh = targets[x]
                elif x in out.keys():
                    xh = out[x]
                #don't check y in input. 
                if y in targets.keys():
                    yh = targets[y]
                elif y  in out.keys():
                    yh = out[y]
                else:
                    raise TypeError("didn't have right format in pairs")
                #print(f"xh: ", xh.shape)
                #print(f"yh: {yh.shape}")
                val = fn(xh, yh)
                #print(F"val: {val}")
                key = f"{x}_wrt_{y}"
                if outputDict.get(f"{key}/{suffix}", None) is None: 
                    #print("inside")
                    outputDict[f"{key}/{suffix}"] = 0
                outputDict[f"{key}/{suffix}"] +=val
                #print("here")
            except Exception as e:
                print(f"[JacobianDiag] Failed for {pair}/{suffix}: {e}")
    return outputDict

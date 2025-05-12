#GLOBAL VALUE. 
from diagnostics.utils import generate_fieldnames
#TODO: Create more diagnostics. Add to config. 
DIAGNOSTIC_REGISTRY = {}
DIAGNOSTIC_FIELD_REGISTRY= {}

def register_diagnostic(name = None, fields = None):
    def decorator(fn):
        diag_name = name or fn.__name__
        print("register diag name: ",  diag_name)
        DIAGNOSTIC_REGISTRY[diag_name] = fn
        if fields is None: 
            DIAGNOSTIC_FIELD_REGISTRY[diag_name] = lambda cfg:  default_fields_from_config(diag_name, cfg)
        elif callable(fields):
            DIAGNOSTIC_FIELD_REGISTRY[diag_name] = fields 
        else:
            DIAGNOSTIC_FIELD_REGISTRY[diag_name] = lambda cfg: fields
        return fn
    return decorator

def get_diagnostics():
    return DIAGNOSTIC_REGISTRY

def get_fields():
    return DIAGNOSTIC_FIELD_REGISTRY

def default_fields_from_config(diag_name, config):
    cfg = config.get("diagnostics_config", {})
    keys = cfg.get(f"{diag_name}_keys", [])
    suffixes = cfg.get(f"{diag_name}_suffixes", [])
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(suffixes, str): suffixes = [suffixes]
    return generate_fieldnames(keys, suffixes)
    
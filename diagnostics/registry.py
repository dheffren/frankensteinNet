#GLOBAL VALUE. 

#TODO: Create more diagnostics. Add to config. 
DIAG_REGISTRY = {}
DIAG_FIELDS = {}

def register_diagnostic(name = None, field_fn = None):
    #If field_fn omitted assumes diagnostic will "discover" it's own fields at runtime. 
    def decorator(fn):
        diag_name = name or fn.__name__
        
        DIAG_REGISTRY[diag_name] = fn
        DIAG_FIELDS[diag_name] = field_fn or default_field_fn(diag_name)
        return fn
    return decorator

def get_diagnostics():
    return DIAG_REGISTRY
def get_diagnostic(name):
    return DIAG_REGISTRY[name]
def get_fields():
    return DIAG_FIELDS
def default_field_fn(diag_name):
    """
    TODO: Check if this works properly. 
    """
    def fn(cfg:dict) -> list[str]:
        dcfg = cfg.get("diagnostics_config", {})
        
        #case 1: Explicit list <diag_name>_fields
        explicit = dcfg.get(f"{diag_name}_fields")
        if explicit is not None:
            return explicit if isinstance(explicit, list) else [explicit]
        
        # Case 2: keys + suffixes  (<diag_name>_keys,  _suffixes)
        keys     = dcfg.get(f"{diag_name}_keys")
        suffixes = dcfg.get(f"{diag_name}_suffixes")
        if keys is not None and suffixes is not None:
            if isinstance(keys, str):      keys     = [keys]
            if isinstance(suffixes, str):  suffixes = [suffixes]
            #returns "field name"
            return [f"{k}/{s}" for k in keys for s in suffixes]
        
        #Nothing declared - no fixed header, let logger discover. 
        return []
    return fn

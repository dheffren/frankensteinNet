#GLOBAL VALUE. 
DIAG_REGISTRY = {}

def register_diagnostic(name, default_trigger, default_every, priority):
    #If field_fn omitted assumes diagnostic will "discover" it's own fields at runtime. 
    def decorator(fn):
        diag_name = name or fn.__name__
        DIAG_REGISTRY[diag_name] = {"fn": fn, "trigger": default_trigger, "every":default_every}
        return fn
    return decorator

def get_diagnostics():
    return DIAG_REGISTRY
def get_diagnostic(name):
    return DIAG_REGISTRY[name]

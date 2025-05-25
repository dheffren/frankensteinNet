HOOK_REGISTRY = {}

def register_hook(name, default_trigger = "epoch", default_every=1):
    def decorator(fn):
        print(f"[HookRegistry] Registered dataset: {name}")
        HOOK_REGISTRY[name] = {"fn": fn, "trigger": default_trigger, "every":default_every}
        return fn
    return decorator

def get_registered_hook(name):
    return HOOK_REGISTRY[name]
def get_hooks():
    return HOOK_REGISTRY
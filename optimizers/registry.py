OPTIMIZER_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        print(f"[OptimizerRegistry] Registered optimizer: {name}")
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator

def get_registered_optimizer(name):
    return OPTIMIZER_REGISTRY[name]
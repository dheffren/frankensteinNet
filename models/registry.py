MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        print(f"[DatasetRegistry] Registered dataset: {name}")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_registered_model(name):
    return MODEL_REGISTRY[name]
DATASET_REGISTRY = {}

def register_dataset(name):
    def decorator(cls):
        print(f"[DatasetRegistry] Registered dataset: {name}")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_registered_dataset(name):
    return DATASET_REGISTRY[name]
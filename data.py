import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
"""
Note: Write a custom method for each potential "type" of dataset i have. 
Then call this from the utils with the data path specified. 
TODO: Get this method to work with loaded data, make sure it works period. See where this is called in setup. 
"""
def get_dataloaders(config, train = True):
    cfg = config["data"]
    full_dataset = get_dataset(config, train)
    val_len = int(cfg["val_split"] * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    return {"train": train_loader, "val":val_loader}, full_dataset.get_metadata()
def get_dataset(config, train = True):
    dataset_name = config["data"]["dataset"]
    print(f"dataset_name: {dataset_name}")
    path = config["data"]["path"]
    transform = get_transforms(config, train = train)
    from datasets.data_registry import get_registered_dataset
    DatasetClass = get_registered_dataset(dataset_name)
    #should I specify a path to download FROM vs root? 
    return DatasetClass(root = path, train = train, transform = transform)
    """
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST(root=path, train=train, download=True, transform=transform)
    elif dataset_name  == "CIFAR10":
        full_dataset = datasets.CIFAR10(root=path, train=train, download=True, transform=transform)
    elif dataset_name == "customDataset":
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return full_dataset
    """
def get_transforms(config, train = True):
    #THIS IS THE PROBLEM. 
    tfm_cfg = config.get("transform_config", {})
    if train:
        return transforms.Compose([
            #transforms.Resize(tfm_cfg.get("resize", 28)),
            transforms.ToTensor(),
            transforms.Normalize(*tfm_cfg.get("normalize", [0.5, 0.5]))
        ])
    else:
        return
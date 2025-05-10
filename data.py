import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
"""
Note: Write a custom method for each potential "type" of dataset i have. 
Then call this from the utils with the data path specified. 
TODO: Get this method to work with loaded data, make sure it works period. See where this is called in setup. 
"""
def get_dataloaders(config):
    
    cfg = config["data"]
    
    transform_list = [transforms.ToTensor()]
    if cfg.get("normalize", False):
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    transform = transforms.Compose(transform_list)

    if cfg["dataset"] == "MNIST":
        full_dataset = datasets.MNIST(root=cfg["path"], train=True, download=True, transform=transform)
    elif cfg["dataset"] == "CIFAR10":
        full_dataset = datasets.CIFAR10(root=cfg["path"], train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")

    val_len = int(cfg["val_split"] * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    return train_loader, val_loader
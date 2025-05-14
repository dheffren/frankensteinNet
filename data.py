import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from normalize import Normalizer
from pathlib import Path
import yaml

"""
Note: Write a custom method for each potential "type" of dataset i have. 
Then call this from the utils with the data path specified. 
TODO: Get this method to work with loaded data, make sure it works period. See where this is called in setup. 
"""
def get_dataloaders(config, train = True):
    cfg = config["data"]
    full_dataset, found, mean, std = get_dataset(config, train)
    val_len = int(cfg["val_split"] * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    if not found: 
        mean, std = compute_mean_std(train_loader)
        save_normalization_stats(mean, std,cfg["path"])
        #now data isn't loaded with normalizations - need to redo. 
        #NOTE: The "different thing" is that we saved a file outside of the program. 
        return get_dataloaders(config, train = True)
    metadata = full_dataset.get_metadata()
    #need to complete logic.
    normalizer = Normalizer(mean, std, mode="zscore")
    metadata["normalizer"] = normalizer
    return {"train": train_loader, "val":val_loader}, metadata

def get_dataset(config, train = True):
    #TODO: Configure this normalization stuff to work in the general case: with multiple inputs or whatever. 
    dataset_name = config["data"]["dataset"]
    print(f"dataset_name: {dataset_name}")
    path = config["data"]["path"]
    found = True
    mean = None
    std = None
    try:
        mean, std = load_normalization_stats(path)
        transform = get_transforms(config, train = train, apply_normalization=True, mean=mean, std=std)
    except Exception as e:
        transform = get_transforms(config, train = train)
        found = False
    from datasets.data_registry import get_registered_dataset
    DatasetClass = get_registered_dataset(dataset_name)
    #should I specify a path to download FROM vs root? 
    return DatasetClass(root = path, train = train, transform = transform), found, mean, std

def get_transforms(config, train = True, apply_normalization = False, mean = None, std = None):
    #THIS IS THE PROBLEM. 
    #TODO: Adjust this to properly normalize the input images. 
    tfm_cfg = config.get("transform_config", {})
    tfms = [transforms.ToTensor()]
    if apply_normalization: 
        tfms.append(transforms.Normalize(mean, std))
    #add additional augmentations here. 
    if train:
        pass
    else: 
        pass
    return transforms.Compose(tfms)

def load_normalization_stats(dataset_root):
    path = Path(dataset_root)/ "normalization.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No normalization.yaml found in {dataset_root}")
    with open(path, "r") as f:
        stats = yaml.safe_load(f)
    if not ("mean" in stats and "std" in stats):
        raise ValueError(f"normalization.yaml exists but is missing keys: {stats}")
    return stats["mean"], stats["std"]

def save_normalization_stats(mean, std, path):
    name = "normalization.yaml"
    pathL = Path(path) / name
    with open(pathL, "w") as f:
        yaml.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
def compute_mean_std(loader):
    mean = 0.
    std = 0.
    n = 0
    for imgs, *_ in loader:
        imgs = imgs.view(imgs.size(0), imgs.size(1), -1)  # flatten H,W
        mean += imgs.mean(2).sum(0)
        std  += imgs.std(2).sum(0)
        n += imgs.size(0)
    return mean / n, std / n
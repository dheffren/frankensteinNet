import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from normalize import Normalizer, build_normalizers
from pathlib import Path
import yaml
from transformations import TransformRegistry, build_transforms

"""
Note: Write a custom method for each potential "type" of dataset i have. 
Then call this from the utils with the data path specified. 
TODO: Get this method to work with loaded data, make sure it works period. See where this is called in setup. 
"""
def get_dataloaders(config, train = True):
    cfgD = config["data"]
    prepare_dataset(cfgD) #prepares the mean and std for normalization.  
    path = cfgD["path"]
    
    stats = load_normalization_stats(path) # Dictionary with mean, std. 
    
    transforms = build_transforms(cfgD, stats) #transform Registry object. 
    full_dataset = get_dataset(cfgD, train, transforms)
    val_len = int(cfgD["val_split"] * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=cfgD["batch_size"], shuffle=cfgD["shuffle"], num_workers=cfgD["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfgD["batch_size"], shuffle=False, num_workers=cfgD["num_workers"])

    metadata = full_dataset.get_metadata()
    #need to complete logic.
    normalizer = build_normalizers(stats)
    metadata["normalizer"] = normalizer
    #TODO: Make normalizer work with the rest of the project.  
    return {"train": train_loader, "val":val_loader}, metadata
def prepare_dataset(cfgD): #config["data"]
    path = cfgD["path"]
    try: 
        stats = load_normalization_stats(path) #won't return empty dictionary - may return empty dict values. 
        validate_normalization_stats(stats, cfgD)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"[INFO] Invalid or missing normalization info: {e}")
        raw_ds = get_dataset(cfgD, transform=None)
        stat_loader = torch.utils.data.DataLoader(
            raw_ds, 
            batch_size = cfgD["batch_size"], 
            shuffle = False, 
            num_workers = cfgD["num_workers"])
        #compute ONLY for image keys. 
        #TODO: input right keys. It seems like it naturally checks that it only does the right ones? But are we sure? 
        #TODO: I think this is BACKWARD. - the alt one is backwards. Check if other one works. 
        keys = [k for k, tlist in cfgD["transforms"].items() if "Normalize" in tlist]
        stats = compute_mean_std_for_keys(stat_loader, keys = keys)
        print(stats)
        #save mean and std to disk. 
        save_normalization_stats(stats,  path)
    #after this - know it works. Thus, can proceed normally
    return 
def validate_normalization_stats(stats:dict, cfgD:dict):
    #TODO: Should logic for transform keys instead come from the data itself? Ie from where we convert the data via dictionary. 
    transform_keys = cfgD["transforms"].keys() #list of ALL transform keys - not just those with normalization. 
    for key in transform_keys:
        if "Normalize" not in cfgD["transforms"][key]: continue # Check if normalize is in the list of transforms for that given key.  
        if key not in stats["mean"] or key not in stats["std"]: #throw error, this shouldn't happen if file is correct. 
            raise ValueError(f"Missing mean/std for key '{key}' in metadata.")
        mean = stats["mean"][key]
        std  = stats["std"][key]
        if not isinstance(mean, list) or not isinstance(std, list):
            raise TypeError(f"Mean/std for '{key}' should be lists, got {type(mean)} / {type(std)}")
        if len(mean) != len(std):
            raise ValueError(f"Mean/std for '{key}' must have same length: {mean} vs {std}")
        #check value ranges. 
        if not all(std>0): 
            raise ValueError("Std at or below 0")
        #check shape matches with dataset. 


def get_dataset(cfgD, train = True, transform = None): 
    dataset_name = cfgD["dataset"]
    path = cfgD["path"]
    from datasets.data_registry import get_registered_dataset
    DatasetClass = get_registered_dataset(dataset_name)
    #should I specify a path to download FROM vs root? 
    #TODO: Check class takes in right transformation object. And that it performs transformation right. 
    return DatasetClass(root = path, train = train, transform = transform)


def load_normalization_stats(dataset_root):
    """
    csv data saved as a dict of mean and std, where each of those two can have different "input keys". 
    """
    path = Path(dataset_root)/ "normalization.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No normalization.yaml found in {dataset_root}")
    with open(path, "r") as f:
        stats = yaml.safe_load(f)
    if not ("mean" in stats and "std" in stats):
        raise ValueError(f"normalization.yaml exists but is missing keys: {stats}")
    #Not sure the default save is configured for this. Might have to create manually. 
    if (not isinstance(stats["mean"], dict)) or (not isinstance(stats["std"], dict)):
        raise ValueError(f"normalization.yaml exists but the keys aren't dictionaries. ")
    return stats

def save_normalization_stats(stats, path):
    name = "normalization.yaml"
    pathL = Path(path) / name
    with open(pathL, "w") as f:
        #TODO: Fix this for new layout. .tolist is wrong. 
        yaml.dump({"mean": stats["mean"], "std":stats["std"]}, f)
def compute_mean_std_for_keys(loader, keys, eps = 1e-6):
    """
    Note: Need data to be returned/saved as a dictionary in this case. 
    Keys = JUST keys that we're going to normalize with. 
    """
    stats = {k: {"sum": 0., "sum_sq": 0., "count": 0} for k in keys}
    for batch in loader:
        for k in keys:
            if k not in batch: continue
            x = batch[k]  # shape: [B, C, H, W]
            if not is_image_like(x): continuel
            B, C, H, W = x.shape
            stats[k]["sum"]    += x.sum(dim=(0, 2, 3))  # sum over pixels per channel
            stats[k]["sum_sq"] += (x ** 2).sum(dim=(0, 2, 3))
            stats[k]["count"]  += B * H * W
        means = {}
    stds = {}
    for k in keys:
        s = stats[k]
        mean = s["sum"] / s["count"]
        std_raw = ((s["sum_sq"] / s["count"]) - mean**2).sqrt()
        std = std_raw.clamp(min = eps)
        means[k] = mean.tolist()
        stds[k]  = std.tolist()
    stats = {"mean": means, "std": stds}
    print("stats: ", stats)
    return stats
def compute_mean_std_for_keysAlt(loader, keys):
    #TODO: THIS IS THE WRONG WAY. INSIDE AND OUTSIDE KEYS REVERSED. 
    #Figure out which of these we want. should do normalize_keys = [
    #k for k, tlist in cfg["transforms"].items()
    #if "Normalize" in tlist]
    stats = {}
    for k in keys:
        sums, sqs, count = 0., 0., 0
        for batch in loader:
            if k not in batch: continue
            x = batch[k]
            if not is_image_like(x): continue
            B, C, H, W = x.shape
            sums += x.sum(dim=(0, 2, 3))
            sqs  += (x ** 2).sum(dim=(0, 2, 3))
            count += B * H * W
        if count > 0:
            mean = (sums / count).tolist()
            std = ((sqs / count - (sums / count)**2).sqrt()).tolist()
            stats[k] = {"mean": mean, "std": std}
    return stats
def is_image_like(t: torch.Tensor):

    return t.ndim == 4 or (t.ndim == 3 and t.shape[0] in (1, 3))

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
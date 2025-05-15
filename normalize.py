import torch
class Normalizer:
    #normalizers - Assumes images. Other data idk rn. 
    def __init__(self, mean, std, mode="zscore"):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std  = torch.tensor(std).view(-1, 1, 1)
        self.mode = mode

    def normalize(self, x):
        if self.mode == "zscore":
            return (x - self.mean) / self.std
        elif self.mode == "01":
            return x.clamp(0, 1)
        elif self.mode == "-1_1":
            return x * 2 - 1
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

    def denormalize(self, x):
        if self.mode == "zscore":
            return x * self.std + self.mean
        elif self.mode == "01":
            return x
        elif self.mode == "-1_1":
            return (x + 1) / 2
class NormalizerRegistry:
    """
    Stores per key logic. Only acts on those that NEED normalization. Could do NOTHING each time. 
    """
    def __init__(self, norm_dict: dict[str, Normalizer]):
        self.norms = norm_dict
    def normalize(self, sample_dict: dict):
        return {k: self.norms[k].normalize(v) if k in self.norms else v
                for k, v in sample_dict.items()}

    def denormalize(self, sample_dict: dict):
        return {k: self.norms[k].denormalize(v) if k in self.norms else v
                for k, v in sample_dict.items()}
def build_normalizers(stats) -> NormalizerRegistry:
    """
    Metadata: 
    a dictionary with mean and standard deviation. 
    Each of those is a dict with the keys of all the inputs. (ie the mean or std of each one). 

    """
    mean_dict  = stats["mean"] # in case of no keys with normalize - should just return empty dictionary. Then there'd be no keys. Ie normalizers would be empty dict. 
    std_dict = stats["std"]
    normalizers = {}

    for k in mean_dict: 
        if k in std_dict:  #extra safety - should always be true. 
            normalizers[k] = Normalizer(mean_dict[k], std_dict[k]) #Deal with zscore? - at this point the logic doesn't allow for anything else. 
    return NormalizerRegistry(normalizers)
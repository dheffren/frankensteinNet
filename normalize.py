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
    Does normalization for model inputs and outputs which is passed to reconstruction and visualization methods. 
    The actual data normalization is called via transformRegistry.

    Note: if on different devices, this doesn't work. 
    """
    def __init__(self, norm_dict: dict[str, Normalizer], aliases: dict[str, str] = None):
        self.norms = norm_dict
        self.aliases = aliases or {}
    def _resolve(self, key):
        #if output key: maps to corresponding input key. 
        #if input key, returns itself. 
        #if key not a valid output key, return itself. 
        return self.aliases.get(key, key)
    def normalize(self, sample_dict: dict):
        return {k: self.norms[self._resolve(k)].normalize(v) if self._resolve(k) in self.norms else v
                for k, v in sample_dict.items()}

    def denormalize(self, sample_dict: dict):
        return {k: self.norms[self._resolve(k)].denormalize(v) if self._resolve(k) in self.norms else v
                for k, v in sample_dict.items()}
def build_normalizers(stats, aliases = None) -> NormalizerRegistry:
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
    return NormalizerRegistry(normalizers, aliases)
class TransformRegistry:
    """
    IDEA: Different pipelines per input. 
    Inputs structured as dict. 
    """
    def __init__(self, transform_dict: dict):
        self.transforms = transform_dict

    def __call__(self, sample: dict):
        #takes in sample dict. Returns dictionary where each element you evaluate the transform on that value. 
        return {
            k: self.transforms.get(k, lambda x: x)(v)
            for k, v in sample.items()
        }
from torchvision import transforms

TRANSFORM_LOOKUP = {
    "ToTensor": transforms.ToTensor,
    "Normalize": lambda mean, std: transforms.Normalize(mean, std),
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "CenterCrop": lambda size: transforms.CenterCrop(size),
    # Add more here
}

def build_transforms(cfgD, stats):
    transform_cfg = cfgD["transforms"]
    keys = transform_cfg.keys()  # e.g., "x1", "x2"

    tfm_dict = {}
    #check this method here. 
    for k in keys:
        tfm_list = []
        #for each transformation for a given key: If normalize is there, apply normalize to it. 
        for tfm_name in transform_cfg[k]:
            if tfm_name == "Normalize":
                mean = stats["mean"][k] #Throw error if you can't access the value from that key - it should be there. 
                std = stats["std"][k]
                tfm = TRANSFORM_LOOKUP["Normalize"](mean, std)
            else:
                tfm = TRANSFORM_LOOKUP[tfm_name]()
            tfm_list.append(tfm)

        tfm_dict[k] = transforms.Compose(tfm_list)

    return TransformRegistry(tfm_dict)
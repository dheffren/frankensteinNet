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
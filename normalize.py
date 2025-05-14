import torch
class Normalizer:
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

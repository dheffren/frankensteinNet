from .data_registry import register_dataset
import torch
@register_dataset("ballRotating")
class BallRotatingDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform = None):
        self.train = train
        self.transform = transform
        self.data = ...  # Load from file, npy, csv, whatever
        self.targets = ...
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
    def create_data(self):
        return 
    
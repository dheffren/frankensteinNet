from torchvision.datasets import MNIST
from .data_registry import register_dataset
from torchvision import transforms
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose
from torch.utils.data import Dataset
import torch
import os

@register_dataset("ModelNet")
class MyModelNet(Dataset):
    """
    Two methods of dealing with the infinite vs finite dimensional thing. 
    1. Sample max number of points, then sample the finite points from that (deterministically). 
    2. Just do one sample, and rely on seeding/randomness throughout to replicate what we need. 

    If i want to do checkpoints where I estimate the bias at DIFFERENT n, then I would need to make a NEW dataset. 
    Does my code have a way of dealing with multiple validation datasets/evaluation loops? 
    """
    def __init__(self, root, n ,  max, train=True, transform=None, download=True):
        #n doesn't do anything right now. 
        pointCloud = SamplePoints(num = max)
    
        self.modelNet = ModelNet(root, train=True, transform = pointCloud)
        self.dataAmt = len(self.modelNet)
        #sample point clouds immediately from this. Need to model randomness? 
        self.index = self.materialize_to_shards(self.modelNet, root + "/processed")
        self.numPoints = max

        super().__init__()
        
    def get_metadata(self):
    #TODO: Fix this for this type of data
        return {
            "num_points": self.numPoints
        }
    def __len__(self):
        return self.dataAmt
    def materialize_to_shards(self, pyg_ds, out_dir):
        #TODO: Make sure this data is deleted once the run is over no? 
        os.makedirs(out_dir, exist_ok=True)
        index = []
        #initial randomness. 
        base_seed = torch.initial_seed()

        for i in range(len(pyg_ds)):
         
            with torch.random.fork_rng():
                torch.manual_seed(base_seed + i)
                data = pyg_ds[i]  # SamplePoints happens here
            item = {
                "inputs": data.pos.cpu().to(torch.float32),               # (N,3)
                "outputs": getattr(data, "y", None).to(torch.float32),     # optional
            }
            path = os.path.join(out_dir, f"{i:07d}.pt")
            torch.save(item, path)
            index.append(path)

        torch.save(index, os.path.join(out_dir, "index.pt"))
        return index
 

    def __getitem__(self, idx):
        #TODO: Check that this is right. 
        #don't apply transform. 
        item = torch.load(self.index[idx], map_location = "cpu")

        return item 
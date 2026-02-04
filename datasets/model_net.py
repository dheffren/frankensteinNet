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
    The way SamplePoints works is each time you call the getitem method you're going to apply the transform BEFORE accessing the data - we don't want this, as 
    we want to have a fixed point cloud during training. 

    Question: Does pre transform run again each tim you call it? WHat i want is I want the modelNet to be saved on my hard drive as the meshses, then the point clouds before the 
    training run. 
    """
    def __init__(self, root, train=True, transform=None, download=True, numPoints = 10000):
        
        pointCloud = SamplePoints(num = numPoints)
    
        self.modelNet = ModelNet(root, train=True, transform = pointCloud)
        self.dataAmt = len(self.modelNet)
        #sample point clouds immediately from this. Need to model randomness? 
        self.index = self.materialize_to_shards(self.modelNet, root + "/processed")
        self.numPoints = numPoints

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
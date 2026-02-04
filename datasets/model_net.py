from torchvision.datasets import MNIST
from .data_registry import register_dataset
from torchvision import transforms
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.transforms import Compose
@register_dataset("ModelNet")
class MyModelNet(ModelNet):
    def __init__(self, root, train=True, transform=None, download=True, numPoints = 10000):
        pointCloud = SamplePoints(num = numPoints)
        if transform is None: 
            transform = pointCloud
        else: 
            transform = Compose([transform, pointCloud])
        super().__init__(root=root, train=train, transform=transform)
        self.input_channels = 1
        
        self.data_shape = (self.input_channels, 28, 28)
        #the transform it expects is a direct transform from torchvision. Our transform is a custom class. So don't pass in that way. 
        self.transformA = transform
    def get_metadata(self):
        return {
            "input_channels": self.input_channels, 
            "input_shape": self.data_shape
        }
    def __getitem__(self, idx):
        #TODO: Check that this is right. 
        #don't apply transform. 
        
        (image, target) = super().__getitem__(idx)
   
        #return data as a dict. 

        element = {"x": image, "y":target}
        #includes normalization. 
        """
        if self.transformA:
            #does this work on tesnors? 
          
            transformedEle = self.transformA(element)
        else:
            #default transformation. 
            transform = transforms.Compose([transforms.ToTensor()])
            transformedEle = {"x": transform(image), "y":target}
        """
        return element
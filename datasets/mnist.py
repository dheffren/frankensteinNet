from torchvision.datasets import MNIST
from .data_registry import register_dataset
@register_dataset("MNIST")
class MNISTDataset(MNIST):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.input_channels = 1
        self.data_shape = (self.input_channels, 28, 28)
    def get_metadata(self):
        return {
            "input_channels": self.input_channels, 
            "input_shape": self.data_shape
        }
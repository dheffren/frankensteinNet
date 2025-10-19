import torch
REPO_DIR = "/mnt/Storage/files/code/dinov3"
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights="/mnt/Storage/files/models/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth")

from .data_registry import register_dataset
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from pathlib import Path
from torchvision import transforms
@register_dataset("ballRotating")
class BallRotatingDataset(torch.utils.data.Dataset):
    #TODO: I have a feeling the root should just be "data/" to be consistent with MNIST, then it should add its own name here. 
    def __init__(self, root = "data/ballRotating", train=True, transform = None, BW = False, img_size= 128, amount = 2000):
        self.train = train
        self.transform = transform
        self.root = Path(root)
        self.BW = BW
        self.input_channels = 3
        self.img_size = img_size
        if self.BW: 
            self.input_channels = 1
        self.dataAmt = amount
        self.data_shape = (self.input_channels, img_size, img_size)
        self.csv_path = self.root/"data.csv"
       
        #TODO: Make it so don't have to generate data every time. 
        self.create_data(output_csv_path = self.csv_path)
        
        self.entries = pd.read_csv(self.csv_path)
    def __len__(self):
        return self.dataAmt

    def __getitem__(self, idx):
      
        #TODO: Add caching or lazy loading or something faster. 
        row = self.entries.iloc[idx]
        x1 = Image.open(row["path1"])
        x2 = Image.open(row["path2"])
        #return data as a dict. 
        element = {"x1": x1, "x2":x2}
        #includes normalization. 
       
        if self.transform:
      
            transformedEle = self.transform(element)
            return transformedEle
        else: 
            #still convert PIL image to tensor. 
            transform = transforms.Compose([transforms.ToTensor()])
             
            element = {"x1": transform(x1), "x2":transform(x2)}
            return element
    def get_metadata(self):

        return {
            "input_channels": self.input_channels, 
            "input_shape": self.data_shape, 
            "latent_dimU": 2, 
            "latent_dimV": 2, 
            "latent_dimC": 2
        }
    def load_data(self):
        
        S1 = np.load(self.root / "sensor1/sensor1.npy")
      
        S2 =np.load(self.root / "sensor2/sensor2.npy")
        return S1, S2
    def make_data(self):
        
 
        sensor1_data = load_images_from_folder(self.root/'sensor1', self.img_size, self.BW)
        sensor2_data = load_images_from_folder(self.root/'sensor2', self.img_size, self.BW)

        print(f"Sensor1 data shape: {sensor1_data.shape}")  # (500, 49152)
        print(f"Sensor2 data shape: {sensor2_data.shape}")


        sensor1 = normalize_images(sensor1_data)
        sensor2 = normalize_images(sensor2_data)

        #not normalized to -1 1, just 0 1. Need to fix. 
        #Fixed the output dim so 128 128 3? Is this right? 
        np.save( self.root/"sensor1/sensor1.npy", sensor1)
        np.save( self.root/"sensor2/sensor2.npy",sensor2)
      
    

    def create_data(self, output_csv_path):
        # 5. Main generation loop
        sensor1_paths = []
        sensor2_paths = []

        # Initialize angles
        angle_common1 = 0
        angle_common2 = 0
        angle_private1 = 0
        angle_private2 = 0

        # Define angular velocities (degrees per snapshot)
        vel_common1 = +3    # Clockwise
        vel_common2 = -3    # Counter-clockwise
        vel_private1 = +7   # Clockwise
        vel_private2 = -5   # Counter-clockwise
        dataCount = self.dataAmt
        imgSize = self.img_size
        with open(output_csv_path, "w", newline="") as csvfile:
            fieldnames = ["path1", "path2"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(dataCount):
                img1, img2 = generate_frame(angle_common1, angle_common2, angle_private1, angle_private2, imgSize)
                # can save additional image metadata like the angle velocity or anything else. 
                path1 = self.root / f"sensor1/sample_{idx:04d}.png"
                path2 = self.root / f"sensor2/sample_{idx:04d}.png"
                
                save_png(img1, path1)
                save_png(img2, path2)

                sensor1_paths.append(path1)
                sensor2_paths.append(path2)
                writer.writerow({"path1": path1, "path2": path2})

                # Update angles
                angle_common1 = (angle_common1 + vel_common1) % 360
                angle_common2 = (angle_common2 + vel_common2) % 360
                angle_private1 = (angle_private1 + vel_private1) % 360
                angle_private2 = (angle_private2 + vel_private2) % 360

        print("Finished generating and saving 500 structured samples.")

        # 6. Visualize first 5 pairs
        visualize_pairs_png(sensor1_paths, sensor2_paths, num_pairs=5)
# 1. Load color images
def load_images_from_folder(folder, img_size, BW = False):
    #TODO: add a "try" here to check if data created yet. 
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            #BW here. 
            if BW: 
                img = Image.open(img_path).resize((img_size,img_size)).convert("L")
            else:
                img = Image.open(img_path).resize((img_size,img_size))
            img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
            images.append(img)  # Flatten to 1D
    if BW:
        return np.array(images)[:, np.newaxis, :, :]
    else:
        return np.permute_dims(np.array(images), (0, 3, 1, 2))  
def normalize_images(images):
    images = images.astype(np.float32)
    return (images - images.min()) / (images.max() - images.min()) * 2 - 1
      
# 1. Circular motion position calculator
def circular_motion(center, radius, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    x = center[0] + radius * np.cos(angle_rad)
    y = center[1] + radius * np.sin(angle_rad)
    return int(x), int(y)

# 2. Generate a single frame given angles
def generate_frame(angle_common1, angle_common2, angle_private1, angle_private2, img_size=256):
    # Create blank images
    colorWhite = (255, 255, 255)
    colorBlack = (0,0,0)
    img1 = Image.new('RGB', (img_size, img_size), colorWhite)
    img2 = Image.new('RGB', (img_size, img_size), colorWhite)
    draw1 = ImageDraw.Draw(img1)
    draw2 = ImageDraw.Draw(img2)

    # Define centers
    global_center = (img_size // 2, img_size // 2)
    inner_radius = 20   # common object motion radius (increased)
    outer_radius = 45   # private object motion radius (increased)

    # Common shape (Blue Circle) positions
    common_center1 = circular_motion(global_center, inner_radius, angle_common1)
    common_center2 = circular_motion(global_center, inner_radius, angle_common2)
    common_size = 10 # radius (fixed)

    # Draw common object
    draw1.ellipse([common_center1[0]-common_size, common_center1[1]-common_size,
                   common_center1[0]+common_size, common_center1[1]+common_size], fill=(0,0,255))

    draw2.ellipse([common_center2[0]-common_size, common_center2[1]-common_size,
                   common_center2[0]+common_size, common_center2[1]+common_size], fill=(0,0,255))

    # Private shapes
    private_center1 = circular_motion(global_center, outer_radius, angle_private1)
    private_center2 = circular_motion(global_center, outer_radius, angle_private2)
    private_size = 10  # half-side length for square, or triangle size

    # Sensor 1 private (Red Square)
    draw1.rectangle([private_center1[0]-private_size, private_center1[1]-private_size,
                     private_center1[0]+private_size, private_center1[1]+private_size], fill=(255,0,0))

    # Sensor 2 private (Green Triangle)
    draw2.polygon([
        (private_center2[0], private_center2[1]-private_size),
        (private_center2[0]-private_size, private_center2[1]+private_size),
        (private_center2[0]+private_size, private_center2[1]+private_size),
    ], fill=(0,255,0))

    return img1, img2

# 3. Save image as PNG
def save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

# 4. Visualize first 5 pairs
def visualize_pairs_png(img_paths1, img_paths2, num_pairs=5):
    fig, axs = plt.subplots(num_pairs, 2, figsize=(6, 3*num_pairs))
    if num_pairs == 1:
        axs = np.expand_dims(axs, axis=0)  # ensure 2D array

    for i in range(num_pairs):
        img1 = np.array(Image.open(img_paths1[i]))
        img2 = np.array(Image.open(img_paths2[i]))
        axs[i,0].imshow(img1)
        axs[i,0].set_title(f"Sensor 1 - Sample {i}")
        axs[i,0].axis('off')

        axs[i,1].imshow(img2)
        axs[i,1].set_title(f"Sensor 2 - Sample {i}")
        axs[i,1].axis('off')

    plt.tight_layout()
    os.makedirs("samples", exist_ok=True)
    plt.savefig("samples/first_5_pairs.png", dpi=300)
    plt.close()
    print("Saved first 5 pairs visualization at samples/first_5_pairs.png")

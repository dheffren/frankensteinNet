# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "torch", "Pillow", "matplotlib", "scikit-learn", "torchvision", "PyYAML"]
# ///
import yaml
from utils.setup import build_model, build_optimizer, build_scheduler, build_dataloaders
from train import Trainer
from runManager import RunManager
from logger import Logger
from pathlib import Path

import argparse
import random
import numpy as np
import torch
"""
This function is called for each "run of the model."
Sweep methods may call this multiple times. 
This is calling things for a SPECIFIC config. 
"""
def set_seed(seed):
    #actually sets the seed for all of this randomness. So it's reproducible. 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=bool, default = False)
    #TODO: add other command line argument parameters here! 
    return parser.parse_args()

args = parse_args()

# Load config
with open("configs/" + args.config) as f:
    config = yaml.safe_load(f)

if args.seed is not None: 
    config["seed"] = args.seed

#set run name
if args.run_name is not None: 
    config["run_name"] = args.run_name
#if don't specify a name
elif "run_name" not in config or config["run_name"] is None:
    config["run_name"]= "temp"
#do other arg parameters here: 
#TODO: Check other params here. 

#Sets the seed for everything to follow. 
set_seed(config["seed"])
# Build components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = str(device)
model = build_model(config).to(device)

optimizer = build_optimizer(model, config)
scheduler = build_scheduler(optimizer, config)
train_loader, val_loader = build_dataloaders(config)
runManager = RunManager(config, "runs", args.resume)
#don't love this reference here. 
logger = Logger(runManager.run_dir, config)

# Train
trainer = Trainer(model, optimizer, scheduler, (train_loader, val_loader), logger, config)
trainer.train()

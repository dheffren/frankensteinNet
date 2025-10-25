# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "Pillow", "matplotlib", "scikit-learn", "torchvision", "PyYAML", "pandas", "seaborn", "wandb", "hessian_eigenthings@git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings"]
# ///
import yaml
from core.setup import  setup_experiment
from core.train import Trainer
from core.runManager import RunManager
from core.logger import Logger
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
    parser.add_argument("--epochs", type=int, default = None)
    return parser.parse_args()

args = parse_args()

# Load config
if args.config[0] == '/':
    path = args.config
else:
    path = "configs/" + args.config
with open(path) as f:
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
if args.epochs is not None:
    config["training"]["epochs"] = args.epochs

#Sets the seed for everything to follow. 
set_seed(config["seed"])
# Build components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = str(device)

#Note: training data affects config. 
bundle = setup_experiment(config)
# Train
trainer = Trainer(bundle.model, bundle.optimizer, bundle.scheduler, bundle.dataloaders, bundle.logger, bundle.hook_manager, bundle.metadata, config)
trainer.train()

bundle.hook_manager.call( trigger = "post_run", trigger_point = 1, run_dir = bundle.run_manager.run_dir
                                   )



import os
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.run_utils import get_run_dir
from datetime import datetime
from pathlib import Path
import yaml
import sys
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
#this is a bit sloppy. 
def redirect_stdout_stderr(run_dir):
    sys.stdout = open(run_dir / "stdout.log", "w")
    sys.stderr = open(run_dir / "stderr.log", "w")

class Logger:
    """
    Use this to save and keep track of all versions, metrics and other details, including: 
    Metrics:
        Train/val loss (numbers) per epoch
        Accuracy/learning rate
        Jacobian norm/sharpness
        Hessian/curvature. 
    Visuals: 
        Loss/validation plot
        Reconstructions ie input output pairs
        Sharpness curves
        PCA/tSNE or other visuals of latent representation
        Project loss landscape
        Weight trajectory PCA 
        Jacobian norm/fisher information heatmap
    Data Snapshots: 
        Latent codes per epoch
        model weights at milestones
        Gradients/jacobians. 
    Plots
    Version
    Config - hyperparameter, seed, losses, optimizer, etc. 
    Diagnostics - computations to probe model behavior (after or during training) 
    metrics - scalar values to track progress and performance - save with "log scalar) 
    Artifacts = Raw data you save from a run - files or arrays. Model weights, latent vectors, reconstructions, diagnostic figures, tSNE plot. 
    Saved with logger save checkpoint, array, and plot. 
    Checkpoints
    All saved plots, metrics, versions, seed, config, hyperparameters, diagnostic artifacts, and checkpoints go through th elogger. 
    """
    def __init__(self, config,  use_wandb = True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
      
        self.run_dir, self.run_name = get_run_dir(base_dir = "runs", base_name = self.run_name, naming = config.get("run_naming", "index"))
        
        config["run_name"] = self.run_name #update the run name to be the name + number or whatever thing i use. 
    
        self._init_csv_logger()
       
        #save the config for the run. 
        self._save_config(config)
        #saves standard output and error. 
        redirect_stdout_stderr(self.run_dir)
        #TODO: Fix this - really annoying!!!
        self.field_names = ["step", "train_loss", "val_loss", "train/loss", "lr"]
        if self.use_wandb:
            wandb.init(project=config.get("project_name", "default_project"),
                       name=self.run_name,
                       config=config)

    def _init_csv_logger(self):
        #maybe check for overwrites? 
        self.csv_path = self.run_dir/ "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None  # Initialized on first log

    def _save_config(self, config):
        """
        Save config file in runs/config_name as both yaml (editable) and json (wandb). 
        
        Bonus: Check what to do for overwriting. 
        """
        #path where the runs are saved. Folder runs/runName/ stuff here
        config_path = self.run_dir / "config.yaml"
        if config_path.exists():
            #print(f"⚠️ Config already exists at {config_path}, overwriting.")
            raise FileExistsError(f"Run config already exists at {config_path}")
        with open(config_path, "w") as f:
            #could write something checking for overwriting/collision. 
            yaml.dump(config, f)
        #don't need to check json, config should already exist. But Potential source of error here. 
        with open(self.run_dir/"config.json", "w") as f:
            json.dump(config, f, indent = 2)

    #TODO: Fix this so each epoch is ONE row - can also change scope. 
    def log_scalar(self, name, value, step):
        #log scalar saves the values of a given scalar. 
        if self.csv_writer is None:
            #need to preregister metrics up front. 
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.field_names)
            self.csv_writer.writeheader()
        self.csv_writer.writerow({"step": step, name: value})
        self.csv_file.flush()

        if self.use_wandb:
            wandb.log({name: value}, step=step)

    def save_plot(self, fig, name):
        # in the plots subfolder. 
        plot_path = self.run_dir / "plots"
        plot_path.mkdir(parents=True, exist_ok = True)
        path = plot_path / name
        fig.savefig(path)
        plt.close(fig)
        if self.use_wandb:
            wandb.log({name: wandb.Image(path)})

    def save_checkpoint(self, model, epoch):
        check_path = self.run_dir / "checkpoints"
        check_path.mkdir(parents=True, exist_ok = True)

        path = check_path/f"model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), path)
        if self.use_wandb:
            wandb.save(path)

    def save_array(self, array, name):
        path = self.run_dir / f"{name}.npy"
        np.save(path, array)
        if self.use_wandb:
            wandb.save(path)

    def close(self):
        self.csv_file.close()
        if self.use_wandb:
            wandb.finish()
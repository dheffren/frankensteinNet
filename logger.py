import os
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from diagnostics import get_diagnostics, get_fields
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
    def __init__(self, run_dir, config,  use_wandb = True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        #initial run name
        self.run_dir = run_dir
        #update run dir as we go. Automatically deals with repeat names. 
        #self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        self.project = config.get("logging", {}).get("project", "default")
        self.run_name = config.get("run_name", "unnamed_run")

        self._current_metrics = {}
        self._last_step = None
        #TODO: Fix this - really annoying!!! - Update - fixed, new registry system. 
        self.field_names = self._collect_all_fieldnames(config)
       
        self._init_csv_logger()
        #saves standard output and error. 
        redirect_stdout_stderr(self.run_dir)
  
        if self.use_wandb:
            wandb.init(project=self.project,
                       name=self.run_name,
                       config=config)
                       
    def _collect_all_fieldnames(self, config):
        #this LOOKS right. However, I'm not sure the COMPUTATIONS are right. 
        #TODO: Manually set order, get rid of "sort". 
        field_names = {"step", "train/loss", "val/loss", "lr"}
        active_diags = config.get("diagnostics", [])
        print("Active diags: ", active_diags)
        field_registry = get_fields()
        print("Field registry: ", field_registry)
        for diag_name in active_diags:
            print("Diag name: ", diag_name)
            #what does this return if doesn't find it? 
            get_fields_fn = field_registry.get(diag_name)
            if get_fields_fn is not None:
                 # Call diagnostic's fieldname generator (lambda or custom)
                new_fields = get_fields_fn(config)
                if isinstance(new_fields, str):
                    new_fields = [new_fields]
                field_names.update(new_fields)
            else: 
                print(f"[Warning] Diagnostic '{diag_name}' has no field generator — skipping fields.")
        #sorted how????
        return sorted(field_names)
    def _init_csv_logger(self):
        #maybe check for overwrites? 
        self.csv_path = self.run_dir/ "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames = self.field_names)
        self.csv_writer.writeheader()

    def log_scalar(self, name, value, step):
        #flush manually in the trainer loop. 
        self._current_metrics[name] = value
        self._last_step = step
        
    def _flush_metrics(self, step):
        row = {"step": step}
        row.update(self._current_metrics)
        print("current metrics: ", self._current_metrics)
        #if i don't fill in all metrics, this throws an error, thinking it is a set. 
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        if self.use_wandb:
            wandb.log(row, step=step)

    def flush(self, step=None):
        step = step or self._last_step
        if step is None: 
            return
        self._flush_metrics(step)
        self._current_metrics = {}
        self._last_step = None
    def _update_writer_if_needed(self, new_key):
        #TODO: Make sure this method works ok. - DONT NEED ANYMORE, get all fieldnames at start properly. 
        #Dynamically update field names - don't use unless absolutely necessary. 
        if new_key in self.field_names:
            return
        self.field_names.append(new_key)

        #Rewind the file, reread rows, and rewrite everythying
        self.csv_file.seek(0, os.SEEK_END)
        self.csv_writer = csv.DictWriter(self.csv_file, field_names = self.field_names)
        #DO I :REWRITE HEADER HERE? 
        #Can use buffered dataframe instead. 
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
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
       
        self.field_names = self._collect_all_fieldnames(config)
        print(self.field_names)
        self._rows = []
        self._init_csv_logger()
        #saves standard output and error. 
        redirect_stdout_stderr(self.run_dir)
  
        if self.use_wandb:
            wandb.init(project=self.project,
                       name=self.run_name,
                       config=config)
                       
    def _collect_all_fieldnames(self, config):
        #TODO: this works really well, but the last thing I need to check is if we dynamically add something at a DIFFERENT epoch than 0 - does it rewrite the whole file correctly. 
        field_names = ["step", "train/loss", "val/loss", "lr"]
        active_diags = config.get("diagnostics", [])
        field_registry = get_fields()
      
        for diag_name in active_diags:
            #what does this return if doesn't find it? 
            get_fields_fn = field_registry.get(diag_name)
 
            if get_fields_fn is not None:
                 # Call diagnostic's fieldname generator (lambda or custom)
                new_fields = get_fields_fn(config)
                if isinstance(new_fields, str):
                    new_fields = [new_fields]
                field_names+=[f"{diag_name}/{f}" for f in new_fields]
            else: 
                print(f"[Warning] Diagnostic '{diag_name}' has no field generator — skipping fields.")
        return field_names
       
    def _init_csv_logger(self):
        #maybe check for overwrites? 
        self.csv_path = self.run_dir/ "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames = self.field_names, extrasaction = "ignore") #the  ignore allows dyanamic row addition. 
        self.csv_writer.writeheader()

    def log_scalar(self, name, value, step):
        #if want to REMOVE dynamic addition - remove this if statement and the dictionary will raise a value error. 
        if name not in self.field_names:
            self.field_names.append(name)
            #allow dynamically updating. 
            self.csv_writer.fieldnames.append(name)
            self._rewrite_csv_header()
        #flush manually in the trainer loop. 
        self._current_metrics[name] = value
        self._last_step = step
        
    def _flush_metrics(self, step):
        row = {"step": step}
        row.update(self._current_metrics)
        #Note: This is only for dynamically adding fields/columns to CSV. it allows header to be udpated. 
        self._rows.append(row)

        #if i don't fill in all metrics, this throws an error, thinking it is a set. 
        # DictWriter will drop keys it hasn't seen yet, but because appended them above, it now knows them. 
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
    def _rewrite_csv_header(self):
        """
        Rewinds and rewrites the CSV file with the new header,
        preserving already buffered rows.

        Only for dynamically added column head. 
        """
        
        self.csv_file.seek(0)
        self.csv_file.truncate(0)
        self.csv_writer = csv.DictWriter(self.csv_file,
                                        fieldnames=self.field_names,
                                        extrasaction="ignore")
        self.csv_writer.writeheader()
        for row in self._rows:
            self.csv_writer.writerow(row)
    def save_plot(self, fig, name):
        # in the plots subfolder. 
        plot_path = self.run_dir / "plots"
        plot_path.mkdir(parents=True, exist_ok = True)
        path = plot_path / name
        #hopefully this does what I want. 
        os.makedirs(os.path.dirname(path), exist_ok = True)
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
    def save_artifact(self, array, name):
        path_arr = self.run_dir / "artifacts" 
        
        path_arr.mkdir(parents=  True, exist_ok = True)
        #want to add further subfolder functionality. 
        
        path = path_arr / f"{name}.npy"
        #hopefully this does what I want. 
        os.makedirs(os.path.dirname(path), exist_ok = True)
        np.save(path, array)
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
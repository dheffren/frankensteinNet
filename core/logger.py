import os
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import inspect
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
from typing import Dict, Any
import os, tempfile, io

def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        k2 = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
        if isinstance(v, dict):
            out |= _flatten(v, k2)
        else:
            out[k2] = float(v)
    return out
@dataclass(frozen=True)
class ArtifactContext:
    run_id: str
    epoch: int
    step: int
    trigger: str      # "after_backward", "epoch_end", ...
    hook: str         # "weight_perturb", "hessian", ...
    split: str | None = None     # "train" | "val" | "ood/..." or None


#TODO: Erase this. 
def atomic_write_bytes(path: str, data: bytes):
    # I HATE THIS. 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    with os.fdopen(fd, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
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
    def __init__(self, run_dir, config, meta):
        use_wandb = config.get("logging", {}).get("use_wandb")
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.save_artifacts = config.get("logging", {}).get("save_artifacts", False)
        self.save_checkpoints = config.get("training", {}).get("save_checkpoints", False)
        #initial run name
        self.run_dir = run_dir
        #update run dir as we go. Automatically deals with repeat names. 
        #self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        self.project = config["project_name"] # TODO: Fix this. 
        self.run_name = config["run_name"]


        self._last_step = None
        self.meta = meta
        
        self.field_names = []
        self._buffer_by_step = {}   # step -> {key: value}
        self._meta_by_step   = {}   # step -> {"epoch":..., "phase":..., "trigger":..., "hook":...}
        self._artifact_buf = {}
        self._rows = []
        self._init_csv_logger()
        self._prefix_strategy = "phase/trigger/hook" #could customize in config
        #saves standard output and error. 
        redirect_stdout_stderr(self.run_dir)
        
        if self.use_wandb:
            set_wandb_api_key_from_file()
            wandb.init(project=self.project,
                       name=self.run_name,
                       config=config, settings =wandb.Settings( _disable_stats=True, _disable_meta=True))
            self._original_log = wandb.log
            #wandb.log = self.debug_log
    def format_artifact_path(self, ctx: ArtifactContext, key: str) -> str:
        epst = f"ep{ctx.epoch:04d}-st{ctx.step:09d}"
        split = (ctx.split + "/") if ctx.split else ""
        base, ext = (key.rsplit(".", 1) + [""])[:2]
        ext = f".{ext}" if ext else ""
        return (
                f"{ctx.trigger}/{ctx.hook}/{split}{base}/{epst}{ext}")
    def _prefix_from_ctx(self, ctx) -> str:
        # Options: "phase/trigger/hook", "phase/hook", "hook", etc.
        parts = []
        for token in self._prefix_strategy.split("/"):
            if token == "phase":   parts.append(ctx.phase)               # "train" | "val" | "test"
            elif token == "trigger": parts.append(ctx.trigger)           # e.g., "after_backward", "epoch_end"
            elif token == "hook":    parts.append(getattr(ctx, "hook", ""))  # diag name if present
        return "/".join([p for p in parts if p])
    def flush(self, step):
        #just called once, mostly replaced by log_dict. 
        self._flush_step(step)
    def log_dict(self, ctx, metrics, finalize:bool = False): 
        ### Check the input is the right shape: 
        assert(isinstance(metrics, dict)) #should just be one dictionary, no nesting. 
        # right now assume that metrics is a dictionary of values. 
        prefix = self._prefix_from_ctx(ctx)
        flat = _flatten(metrics, prefix = prefix)


        buf = self._buffer_by_step.setdefault(ctx.step, {})
        buf.update(flat) # Merges buffer. 

        # remember last metadata for this step
        self._meta_by_step[ctx.step] = {
        "epoch": ctx.epoch, "phase": ctx.phase,
        "trigger": ctx.trigger, "hook": getattr(ctx, "hook", None),
        } #last write wins here. 
        #TODO: add json temporary structure. 
        #this is how we flush. 
        if finalize: 
            self._flush_step(ctx.step)
        return 
    def save_plot(self, actx: ArtifactContext, key: str, fig, *, log_key: str | None = None):
        # 1> write png to disk automatically. 
        print("Saving plot")
        plot_path = self.run_dir / "plots"
        plot_path.mkdir(parents=True, exist_ok = True)

        path = plot_path/(self.format_artifact_path(actx, key) + ".png")
        print("path: ", path)
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        atomic_write_bytes(path, buf.getvalue())

        # 2) enqueue for W&B upload at step finalize (optional)
        if self.use_wandb:
            items = self._artifact_buf.setdefault(actx.step, [])
            # wandb.Image is fine for PNGs; log_key controls the chart panel name
            items.append((path, "image", log_key or f"artifact/{actx.hook}/{key}"))
        plt.close(fig)
    def save_artifact(self, actx: ArtifactContext, key: str, data: bytes, *, log_key: str | None = None):
        #FOR NOW: Assuming bytes is a numpy array
        #TODO: Fix this. 
        artifact_path = self.run_dir/"artifacts"
        artifact_path.mkdir(parents = True, exist_ok = True)
        path = artifact_path/self.format_artifact_path(actx, key)
       # if not key.endswith(".npy"):
            #data = np.ascontiguousarray(data).tobytes()
        #else:
        buf = io.BytesIO()
        np.save(buf,data)
        data = buf.getvalue()
        atomic_write_bytes(path, data)
        if self.use_wandb:
            items = self._artifact_buf.setdefault(actx.step, [])
            items.append((path, "file", log_key or f"artifact/{actx.hook}/{key}"))
    
    def _append_csv_row(self, row: dict):
        new_keys = [k for k in row.keys() if k not in self.field_names] # just applied to new metrics. 
        if new_keys: 
            self.field_names.extend(new_keys) # Extend increases the list
            self._rewrite_csv_header() #rewrites the header when adding new things to the field names I think this is breaking? 

        self.csv_writer.writerow(row) #write row to csv
        self._rows.append(row) #use this so can rewrite header. 
        self.csv_file.flush() # put it in csv

    def _flush_step(self, step:int):
        #called from flush. 
        print(f"FLUSHING STEP: {step}")
        flat = self._buffer_by_step.pop(step, None)
        meta = self._meta_by_step.pop(step, {})
        arts = self._artifact_buf.pop(step, [])
        row = {"step":step, **meta, **(flat or {})}
        print("Row: ", row)
        self._append_csv_row(row)
        if self.use_wandb:
            for path, kind, log_key in arts:
                if kind == "image":
                    wandb.log({log_key: wandb.Image(str(path))}, step=step)
                else:
                    wandb.log({log_key: str(path)}, step=step)
            if flat:
                wandb.log(flat, step = step)
            else: 
                wandb.log({}, step=step) # close commit. 
        

    def _init_csv_logger(self):
        #maybe check for overwrites? 
        self.csv_path = self.run_dir/ "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames = self.field_names, extrasaction = "ignore") #the  ignore allows dyanamic row addition. 
        self.csv_writer.writeheader()
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
    def debug_log(self,*args, **kwargs):
        step = kwargs.get("step", "<auto>")
        print(f"[WandB LOG] step={step}, keys={list(args[0].keys()) if args else '??'}")
        
        # Print caller info
        stack = inspect.stack()
        print(f"  Called from: {stack[1].filename}:{stack[1].lineno}")

        return self._original_log(*args, **kwargs)     
    
    def save_checkpoint(self, model, key):
        if not self.save_checkpoints:
            return
        check_path = self.run_dir / "checkpoints"
        check_path.mkdir(parents=True, exist_ok = True)
        
        path = check_path/f"model_{key}.pth"
        torch.save(model.state_dict(), path)
      
        if self.use_wandb:
           

            artifact = wandb.Artifact(name = "model", type = "checkpoint")
            artifact.add_file(path)
            wandb.log_artifact(artifact, aliases = [f"model_{key}.pt"])
    
    def close(self):
        self.csv_file.close()
        if self.use_wandb:
            wandb.finish()
def set_wandb_api_key_from_file(filepath="api.txt"):
    try:
        with open(filepath, "r") as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("WandB API key file is empty.")
            os.environ["WANDB_API_KEY"] = api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find WandB API key file at: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error reading WandB API key: {e}")
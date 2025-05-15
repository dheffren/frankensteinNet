"""
Load metrics .ccsv from runs, merge into a dataframe, compute statistics, generate comparative plots automatically. 
Results Collector/aggregator

TODO: Add more methods here to look at specific statistics or plots. 
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results(pattern="runs/*/metrics.csv"):
    dfs = []
    for path in glob.glob(pattern):
        df = pd.read_csv(path)
        df["run"] = path.split("/")[1]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_loss_curves(metrics_path, save_dir=None, show=False):
    """
    Plots train and validation loss curves from a metrics.csv file.
    """
    metrics = pd.read_csv(metrics_path)
    steps = metrics["step"]

    plt.figure()
    if "train/loss" in metrics.columns:
        plt.plot(steps, metrics["train/loss"], label="Train Loss")
    if "val/loss" in metrics.columns:
        plt.plot(steps, metrics["val/loss"], label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    if save_dir:
        plt.savefig(Path(save_dir) / "loss_curve.png")
    if show:
        plt.show()
    plt.close()
def plot_learning_rate(metrics_path, save_dir=None, show=False):
    """
    Plots learning rate over training.
    """
    metrics = pd.read_csv(metrics_path)
    steps = metrics["step"]
    if "lr" not in metrics.columns:
        return
    print("here")
    plt.figure()
  
    plt.plot(steps, metrics["lr"], label="Learning Rate", color="green")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    if save_dir:
        plt.savefig(Path(save_dir) / "lr_schedule.png")
    if show:
        plt.show()
    plt.close()

def plot_all_metrics(run_dir):
    """
    Loads metrics from the run directory and generates default plots.
    #TODO: Make plot all metrics automatic for all things I'm trying to track. 
    """

    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"

    if not metrics_path.exists():
        print(f"[Analyze] No metrics.csv found in {run_dir}")
        return
    plot_loss_curves(metrics_path, save_dir=run_path)
    plot_learning_rate(metrics_path, save_dir=run_path)
    print(f"[Analyze] Plots saved to {run_path}")

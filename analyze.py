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
def plot_grad_norm_histogram(grad_norms_dict, epoch, level="per_param", save_dir = None, show = False):
    values = list(grad_norms_dict[level].values())
    plt.figure()
    plt.hist(values, bins=30)
    plt.title(f"Gradient Norm Histogram ({level}) - Epoch {epoch}")
    plt.xlabel("Gradient Norm")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "grad_norm_hist.png")
    if show:
        plt.show()
    plt.close()
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
import seaborn as sns
import numpy as np
def plot_grad_norm_heatmap(grad_history, level="per_layer", save_dir = None, show = False):
    """
    grad_history: dict of {epoch: {param_or_layer_name: grad_norm}}
    """
    return 

    
def analyze_grad_norms(metrics_csv_path, save_dir = None, show = False):
    """
    Loads gradient norm metrics from a CSV and plots a histogram (last epoch)
    and a heatmap (over all epochs).

    Args:
        metrics_csv_path (str): Path to the metrics.csv file.
    """
    # Load the CSV
    df = pd.read_csv(metrics_csv_path)

    # Filter to only columns containing 'grad_norm'
    grad_columns = [col for col in df.columns if 'grad_norm' in col]
    print("grad columns: ", grad_columns)
    if not grad_columns:
        raise ValueError("No 'grad_norm' columns found in metrics file.")

    # Get epochs (fallback to index if missing)
    df['epoch'] = df['epoch'] if 'epoch' in df.columns else df.index

    # Melt into long-form
    grad_df = df[grad_columns + ['epoch']]
    melted = grad_df.melt(id_vars='epoch', var_name='param', value_name='grad_norm')

    # === Histogram of last epoch ===
    last_epoch = melted['epoch'].max()
    final_epoch_data = melted[melted['epoch'] == last_epoch]

    plt.figure(figsize=(8, 5))
    plt.hist(final_epoch_data['grad_norm'].dropna(), bins=30)
    plt.title(f"Gradient Norm Histogram (Epoch {last_epoch})")
    plt.xlabel("Gradient Norm")
    plt.ylabel("Parameter Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Heatmap over epochs ===
    heatmap_df = melted.pivot(index='epoch', columns='param', values='grad_norm')
    heatmap_data = heatmap_df.T.fillna(0.0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="magma", cbar=True)
    plt.yticks(fontsize = 2)
    plt.title("Gradient Norms Over Time (per Parameter / Layer / Total)")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter / Layer / Total", fontsize = 10)
    plt.tight_layout()
    plt.show()
    if save_dir:
        plt.savefig(Path(save_dir) / "grad_norm_heatmap.png")
    if show:
        plt.show()
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
    analyze_grad_norms(metrics_path, save_dir = run_path)
    print(f"[Analyze] Plots saved to {run_path}")

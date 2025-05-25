"""
Load metrics .ccsv from runs, merge into a dataframe, compute statistics, generate comparative plots automatically. 
Results Collector/aggregator

TODO: Add more methods here to look at specific statistics or plots. 
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib.colors import LogNorm
import os
import numpy as np
from sklearn.decomposition import PCA

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
def get_method(metrics_path, stringToGet, exclude = None):
    """
    Get all things from metrics.csv with a certain label or note: 
    """
     # Load the CSV
    df = pd.read_csv(metrics_path)

    # Filter to only columns containing 'grad_norm'
    grad_columns = [col for col in df.columns if stringToGet in col]
    if exclude is not None:
        grad_columns = [col for col in grad_columns if exclude not in col]
    
    if not grad_columns:
        raise ValueError(f"No columns containing {stringToGet}found in metrics file.")
    #TODO: Get to deal with epoch vs step. 
    # Get epochs (fallback to index if missing)
    df['epoch'] = df['epoch'] if 'epoch' in df.columns else df.index

   
    #includes epoch and the columns we care about. KNOW epoch and not step here. 
    grad_df = df[grad_columns + ['epoch']]
    return grad_df
def plot_loss_curves(metrics_path, save_dir=None, show=False):
    """
    Plots train and validation loss curves from a metrics.csv file.
    """
    metrics = get_method(metrics_path, "loss")
    plt.figure()
    for col in metrics.columns:
        if col == 'epoch': 
            continue
        plt.plot(metrics["epoch"], metrics[col], label = col)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Losses")
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
    print("got to learning rate")
    metrics = get_method(metrics_path, "lr")
    if "lr" not in metrics.columns:
        return
    print("here")
    plt.figure()
  
    plt.plot(metrics["epoch"], metrics["lr"], label="Learning Rate", color="green")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    if save_dir:
        plt.savefig(Path(save_dir) / "lr_schedule.png")
    if show:
        plt.show()
    plt.close()

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
    print("GOT TO GRAPHING GRAD NORMS. ")
    grad_df = get_method(metrics_csv_path, "grad_norm", exclude = ".")
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
    if save_dir:
        plt.savefig(Path(save_dir) / "grad_norm_hist.png")
    if show:
        plt.show()
    plt.clf()
    # === Heatmap over epochs ===
    heatmap_df = melted.pivot(index='epoch', columns='param', values='grad_norm')
    heatmap_data = heatmap_df.T.fillna(0.0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="magma", cbar=True, norm = LogNorm(vmin =heatmap_df[heatmap_df>0].min().min(), vmax = heatmap_df.max().max()))
    plt.yticks(fontsize = 6)
    plt.title("Gradient Norms Over Time (per Parameter / Layer / Total)")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter / Layer / Total", fontsize = 10)
    plt.tight_layout()
 
    if save_dir:
        plt.savefig(Path(save_dir) / "grad_norm_heatmap.png")
    if show:
        plt.show()
def analyze_weight_norms(metrics_csv_path, save_dir = None, show = False):
    weight_df = get_method(metrics_csv_path, "grad_norm", exclude = ".")
    melted = weight_df.melt(id_vars='epoch', var_name='param', value_name='weight_norm')

    # Lineplot of weight norm over time
    plt.figure(figsize=(10, 6))
    for name, group in melted.groupby('param'):
        plt.semilogy(group['epoch'], group['weight_norm'], label=name)
    plt.title("Weight Norms Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.legend(fontsize=6, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "weight_norm_graph.png")
    if show:
        plt.show()
    plt.clf()
    heatmap_df = melted.pivot(index='param', columns='epoch', values='weight_norm').fillna(0)

    sns.heatmap(heatmap_df, cmap="viridis", cbar=True, norm = LogNorm(vmin =heatmap_df[heatmap_df>0].min().min(), vmax = heatmap_df.max().max()))
    plt.title("Weight Norms Heatmap")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter")
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "weight_norm_heatmap.png")
    if show:
        plt.show()
def load_pca_projections_for_layer(layer_path):
    latents = []
    epochs = []
    layer_path = layer_path + "/projectedExt"
    default_name = "projected_epoch_"
    files = sorted([
        f for f in os.listdir(layer_path)
        if f.startswith(default_name) and f.endswith(".npy")
    ])
    print(files)
    for fname in files:
        epoch_str = fname[len(default_name):-4]
        try:
            print('before this')
            epoch = int(epoch_str)
            print(epoch)
        except ValueError:
            continue  # Skip malformed filenames
      
        path = os.path.join(layer_path, fname)
        print("got path: ", path)
        z = np.load(path)  # shape = (num_samples, latent_dim)
        print(z.shape)
        print("loaded")
        latents.append(z)
        epochs.append(epoch)
    latents_over_time = np.stack(latents, axis=0)
    return latents_over_time, epochs

def load_latent_trajectories_for_layer(layer_path):
    """
    Loads latent vectors from all embed_epoch_{epoch}.npy files inside a given layer directory.
    Returns:
        latents_over_time: np.ndarray, shape (num_epochs, num_samples, latent_dim)
        epochs: List[int]
    """
    files = sorted([
        f for f in os.listdir(layer_path)
        if f.startswith("embed_epoch_") and f.endswith(".npy")
    ])

    latents = []
    epochs = []

    for fname in files:
        epoch_str = fname[len("embed_epoch_"):-4]
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue  # Skip malformed filenames

        path = os.path.join(layer_path, fname)
        z = np.load(path)  # shape = (num_samples, latent_dim)
        latents.append(z)
        epochs.append(epoch)

    latents_over_time = np.stack(latents, axis=0)
    return latents_over_time, epochs
def plot_adjusted_latent_trajectories(run_path, save_dir, show = False):
    num_points = 10
    artifact_root = run_path/"artifacts"
    if not os.path.isdir(artifact_root):
        raise FileNotFoundError(f"No 'artifacts' directory found in {run_path}")

    layer_names = [
        name for name in os.listdir(artifact_root)
        if os.path.isdir(os.path.join(artifact_root, name))
    ]

    for layer in sorted(layer_names):
        layer_path = os.path.join(artifact_root, layer)
        print(f"\n📦 Analyzing layer: {layer}")
        try:
            latents, epochs = load_pca_projections_for_layer(layer_path)
            print("middle")
            plot_adjusted_latent_trajectoriesHelp(latents, epochs, layer_name=layer, num_points=num_points,save_dir = save_dir)
        except Exception as e:
            print(f"⚠️  Skipping layer {layer} due to error: {e}")
def plot_adjusted_latent_trajectoriesHelp(latents, epochs, layer_name, num_points, save_dir):
    num_epochs, num_samples, latent_dim = latents.shape
    num_points = min(num_points, num_samples)
    # Subset: first `num_points` samples
    latents_subset = latents[:, :num_points, :]  # shape = (epochs, points, dim)
    plt.figure(figsize=(10, 7))
    for i in range(num_points):
        traj = latents_subset[:, i, :]
        #TODO: Fix colors
        plt.scatter(traj[:, 0], traj[:, 1], alpha= .6)
        plt.plot(traj[:, 0], traj[:, 1], alpha= .3)
        #plt.scatter(traj[-1, 0], traj[-1, 1], s=10,)  # Mark end

    plt.title(f"Adjusted latent trajectories for Layer: {layer_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / f"{layer_name}_adj_latent_trajectories.png")

def plot_latent_trajectories(run_path, save_dir = None, show = False):
    num_points = 20
    artifact_root = run_path/"artifacts"
    if not os.path.isdir(artifact_root):
        raise FileNotFoundError(f"No 'artifacts' directory found in {run_path}")

    layer_names = [
        name for name in os.listdir(artifact_root)
        if os.path.isdir(os.path.join(artifact_root, name))
    ]

    for layer in sorted(layer_names):
        layer_path = os.path.join(artifact_root, layer)
        print(f"\n📦 Analyzing layer: {layer}")
        try:
            latents, epochs = load_latent_trajectories_for_layer(layer_path)
            plot_latent_trajectoriesHelp(latents, epochs, layer_name=layer, num_points=num_points,save_dir = save_dir)
        except Exception as e:
            print(f"⚠️  Skipping layer {layer} due to error: {e}")
def plot_latent_trajectoriesHelp(latents_over_time, epochs, layer_name, num_points, save_dir):
    """
    Applies PCA to latent vectors and plots trajectory of fixed samples across epochs.

    Args:
        latents_over_time: (num_epochs, num_samples, latent_dim)
        epochs: List[int]
        layer_name: str, for title
        num_points: int, number of sample trajectories to plot
    """
    num_epochs, num_samples, latent_dim = latents_over_time.shape
    num_points = min(num_points, num_samples)

    # Subset: first `num_points` samples
    latents_subset = latents_over_time[:, :num_points, :]  # shape = (epochs, points, dim)

    # Flatten for PCA: (epochs * points, dim)
    flat_latents = latents_subset.reshape(num_epochs * num_points, latent_dim)

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flat_latents)
    reduced = reduced.reshape(num_epochs, num_points, 2)

    # Plot
    plt.figure(figsize=(10, 7))
    for i in range(num_points):
        traj = reduced[:, i, :]
        plt.scatter(traj[:, 0], traj[:, 1], alpha=0.6)
        #plt.scatter(traj[-1, 0], traj[-1, 1], s=10, c='red')  # Mark end

    plt.title(f"Latent Trajectories for Layer: {layer_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / f"{layer_name}_latent_trajectories.png")


def plot_latent_norms(metrics_path, save_dir = None, show = False):
    weight_df = get_method(metrics_path, "latent_norm", exclude = ".")
    melted = weight_df.melt(id_vars='epoch', var_name='param', value_name='latent_norm')
    print(weight_df)
    # Lineplot of weight norm over time
    plt.figure(figsize=(10, 6))
    for name, group in melted.groupby('param'):
        interpolated = group["latent_norm"].interpolate(method = "linear")
        plt.semilogy(group['epoch'], interpolated, label=name)
    plt.title("Latent Norms Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.legend(fontsize=6, loc='upper right')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "latent_norm_graph.png")
    if show:
        plt.show()
    plt.clf()
    heatmap_df = melted.pivot(index='param', columns='epoch', values='latent_norm').fillna(0)

    sns.heatmap(heatmap_df, cmap="viridis", cbar=True, norm = LogNorm(vmin =heatmap_df[heatmap_df>0].min().min(), vmax = heatmap_df.max().max()))
    plt.title("Latent Norms Heatmap")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter")
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "latent_norm_heatmap.png")
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
    save_dir = run_path/"plots"
    plot_loss_curves(metrics_path, save_dir=save_dir)
    plot_learning_rate(metrics_path, save_dir=save_dir)
    analyze_grad_norms(metrics_path, save_dir = save_dir)
    analyze_weight_norms(metrics_path, save_dir = save_dir)
    plot_latent_norms(metrics_path, save_dir)
    plot_latent_trajectories(run_path, save_dir = save_dir)
    plot_adjusted_latent_trajectories(run_path, save_dir)
    print(f"[Analyze] Plots saved to {run_path}")

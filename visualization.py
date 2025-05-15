import matplotlib.pyplot as plt
import torch
import numpy as np

def make_reconstruction_plot(x, x_recon, epoch, num_images=8, meta = None):
    # Plot original and reconstruction
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    #TODO: See what it looks like. 
    #TODO: This doesn't work now because normalizer should act on the INPUT data structure, not this data structure. ADD Alias mapping or some other cheat method. Store in metadata? 
   
    for i in range(num_images):
        #hopefully this is pass by instance. 
        plot_tensor(axs[0,i], x[i])
        axs[0, i].axis('off')
        plot_tensor(axs[1, i], x_recon[i])
        axs[1, i].axis('off')

    axs[0, 0].set_ylabel("Original", fontsize=12)
    axs[1, 0].set_ylabel("Reconstruction", fontsize=12)

    fig.suptitle(f"Reconstructions — Epoch {epoch}")
    plt.tight_layout()
    return fig
def make_dual_reconstsruction_plot(x1,x1_recon, x2, x2_recon, epoch, num_images= 8, meta = None):
    #already renormalized, all need to do her
    
   # x1 = renormalize_data(x1)
    #x1_recon = renormalize_data(x1_recon)
    #x2 = renormalize_data(x2)
    #x2_recon = renormalize_data(x2_recon)
    fig, axs = plt.subplots(4, num_images, figsize=(num_images * 1.5, 3))
    #TODO: See what it looks like. 
    for i in range(num_images):
        #hopefully this is pass by instance. 
        plot_tensor(axs[0,i], x1[i])
        axs[0, i].axis('off')
        plot_tensor(axs[1, i], x1_recon[i])
        axs[1, i].axis('off')
        plot_tensor(axs[2,i], x2[i])
        axs[2, i].axis('off')
        plot_tensor(axs[3, i], x2_recon[i])
        axs[3, i].axis('off')

    axs[0, 0].set_ylabel("Original X1", fontsize=12)
    axs[1, 0].set_ylabel("Reconstruction X1", fontsize=12)
    axs[2, 0].set_ylabel("Original X2", fontsize=12)
    axs[3, 0].set_ylabel("Reconstruction X2", fontsize=12)
    fig.suptitle(f"Reconstructions — Epoch {epoch}")
    plt.tight_layout()
    return fig
def renormalize_data(x):
    #assume data in -1, 1 range. Not sure if RIGHT but assume lol. 

    return (255*(x)).to(dtype = torch.uint8)
def plot_tensor(ax, img_tensor):
    if img_tensor.shape[0] == 1:
        ax.imshow(img_tensor[0], cmap='gray')
    else:
        ax.imshow(img_tensor.permute(1, 2, 0))  # CHW → HWC for RGB
def plot_pca_scree(n_components, explained_var, cum_var, key):
    fig, ax = plt.subplots()
    ax.plot(np.arange(n_components), explained_var, marker='o', label="Explained Variance")
    ax.plot(np.arange(n_components), cum_var, marker='x', label="Cumulative Variance")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Variance Ratio")
    ax.set_title(f"{key} PCA Scree Plot")
    ax.legend()
    return fig



def plot_pca_component(n_components, latent_dim_weights):
    fig, ax = plt.subplots(figsize=(10, min(6, n_components)))
    im = ax.imshow(latent_dim_weights, aspect='auto', cmap='viridis')
    ax.set_xlabel("Original Latent Dimension")
    ax.set_ylabel("PCA Component")
    ax.set_title("PCA Basis Weights")
    fig.colorbar(im, ax=ax)
    return fig




def plot_pca_2d_scatter(components, all_labels, key):
    fig, ax = plt.subplots()
    if all_labels is not None:
        scatter = ax.scatter(components[:, 0], components[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.7)
        legend = ax.legend(*scatter.legend_elements(), title="Class", loc='best')
        ax.add_artist(legend)
    else:
        ax.scatter(components[:, 0], components[:, 1], s=5, alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{key} PCA Scatter (2D)")
    return fig



def plot_pca_3d_scatter(components, all_labels, key):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if all_labels is not None:
        p = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=all_labels, cmap='tab10', s=5, alpha=0.7)
        legend = ax.legend(*p.legend_elements(), title="Class")
        ax.add_artist(legend)
    else:
        ax.scatter(components[:, 0], components[:, 1], components[:, 2], s=5, alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"{key} PCA Scatter (3D)")
    return fig
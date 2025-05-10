import matplotlib.pyplot as plt
import torch

def make_reconstruction_plot(x, x_recon, epoch, num_images=8):
    # Plot original and reconstruction
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    #TODO: See what it looks like. 
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


def plot_tensor(ax, img_tensor):
    if img_tensor.shape[0] == 1:
        ax.imshow(img_tensor[0], cmap='gray')
    else:
        ax.imshow(img_tensor.permute(1, 2, 0))  # CHW → HWC for RGB

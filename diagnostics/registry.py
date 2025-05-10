from .jacobian import compute_jacobian_norm
from .reconstruction_plot import log_reconstruction_plot
#GLOBAL VALUE. 
#TODO: Create more diagnostics. Add to config. 
DIAGNOSTIC_REGISTRY = {
    "reconstruction_plot": log_reconstruction_plot,
    "jacobian_norm": compute_jacobian_norm,
    #"latent_pca": run_latent_pca,
    #"sharpness": compute_sharpness,
}

from .registry import get_diagnostics, get_fields
import importlib
import os
import pathlib
#TODO: See if this works
def auto_import_all_diagnostics():
    diagnostics_dir = pathlib.Path(__file__).parent
    for file in os.listdir(diagnostics_dir):
        if file.startswith("_") or not file.endswith(".py") or file == "registry.py":
            continue
        module_name = f"{__name__}.{file[:-3]}"  # Strip .py
        importlib.import_module(module_name)

# Force loading of all diagnostics to register themselves
#from . import tensor_stats_diag, latent_variance_diag, jacobian, reconstruction_plot
auto_import_all_diagnostics()
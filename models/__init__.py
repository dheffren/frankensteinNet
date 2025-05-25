import importlib
import os
from pathlib import Path
def auto_import_all_models():
    # __file__ path to the current file. 
    models_dir = Path(__file__).parent
    #print(f"dataset dir: {datasets_dir}")
    for file in os.listdir(models_dir):
        if not file.endswith(".py"):
            continue
        if file.startswith("_")or file in ("model_registry.py", "registry.py"):
            continue
        #__name__ - pythyon variable of the module name of the file currently being run or imported
        module_name = f"{__name__}.{file[:-3]}"
        #print(f"Module_Name: {module_name}")
        importlib.import_module(module_name)
auto_import_all_models()
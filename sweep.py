from copy import deepcopy
import subprocess
import sys
from utils.sweeper import launch_sweep
#newer sweep
if __name__ == "__main__":
    sweep_config = sys.argv[1]  # e.g. "sweep_configs/sweep_latent_lr.yaml"
    launch_sweep(sweep_config)
    
#older sweep method. 
def sweep_seeds(base_config_path, seeds = [1,2,3,4,5]):
    #can alter learrning rate and potentially run in parallel with multiprocessing. 
    for seed in seeds: 
        run_config = deepcopy(load_yaml(base_config_path))
        run_config["seed"] = seed
        run_config["run_name"] = f"{run_config['base_name']}_seed{seed}"
        save_temp_yaml(run_config, "tmp.yaml")
        subprocess.run(["python", "main.py", "--config", "tmp.yaml"])
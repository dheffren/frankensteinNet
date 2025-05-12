import itertools, yaml, os, random
from copy import deepcopy
from pathlib import Path
import subprocess

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='.'):
    result = {}
    for k, v in d.items():
        keys = k.split(sep)
        curr = result
        for key in keys[:-1]:
            curr = curr.setdefault(key, {})
        curr[keys[-1]] = v
    return result

def generate_grid(param_dict):
    keys, values = zip(*param_dict.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def generate_random(param_dict, num_samples):
    keys = list(param_dict.keys())
    for _ in range(num_samples):
        sample = {k: random.choice(param_dict[k]) for k in keys}
        yield sample

def apply_override(config, overrides):
    flat_cfg = flatten_dict(config)
    flat_cfg.update(overrides)
    return unflatten_dict(flat_cfg)

def run_experiment(config, run_name):
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    subprocess.run(["python", "main.py", "--config", str(config_path)])

def launch_sweep(sweep_spec_path):
    with open(sweep_spec_path) as f:
        sweep_spec = yaml.safe_load(f)

    base_config = yaml.safe_load(open(sweep_spec["base_config"]))

    sweep_params = sweep_spec["sweep"]
    method = sweep_spec.get("method", "grid")

    if method == "grid":
        variants = list(generate_grid(sweep_params))
    elif method == "random":
        variants = list(generate_random(sweep_params, sweep_spec["random"]["num_samples"]))
    else:
        raise ValueError(f"Unsupported sweep method: {method}")

    for i, overrides in enumerate(variants):
        config = apply_override(deepcopy(base_config), overrides)
        run_name = f"{Path(sweep_spec_path).stem}_run{i:03d}"
        config["run_name"] = run_name

        # Add bootstraps if enabled
        if sweep_spec.get("bootstrap", {}).get("enabled", False):
            for b in range(sweep_spec["bootstrap"]["num_bootstrap"]):
                bs_config = deepcopy(config)
                bs_config["bootstrap"] = {
                    "enabled": True,
                    "seed": sweep_spec["bootstrap"]["resample_seed_base"] + b
                }
                bs_config["run_name"] += f"_boot{b}"
                run_experiment(bs_config, bs_config["run_name"])
        else:
            run_experiment(config, run_name)
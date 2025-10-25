# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "torch", "Pillow", "matplotlib", "scikit-learn", "torchvision", "PyYAML", "pandas", "wandb", "hessian_eigenthings@git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings", "seaborn"]
# ///
# tools/sweep.py
import argparse, itertools, random, subprocess, yaml, csv, tempfile, datetime
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------
# Dict helpers
# ---------------------------------------------------------------------
def flatten(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, key, sep=sep))
        else:
            items[key] = v
    return items


def unflatten(d, sep="."):
    res = {}
    for k, v in d.items():
        cur = res
        parts = k.split(sep)
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return res


# ---------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------
def grid_variants(param_dict):
    keys, values = zip(*param_dict.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def random_variants(param_dict, n):
    keys = list(param_dict.keys())
    for _ in range(n):
        yield {k: random.choice(param_dict[k]) for k in keys}


# ---------------------------------------------------------------------
# Core sweep launcher
# ---------------------------------------------------------------------
def apply_overrides(base, overrides):
    flat = flatten(base)
    flat.update(overrides)
    return unflatten(flat)

def launch_main(temp_cfg: Path):
    #what does check = True do? 
    subprocess.run(["python", "main.py", "--config", str(temp_cfg)], check = True)


def launch_sweep(spec_path: Path):
    spec = yaml.safe_load(open(spec_path))
    
    base_cfg = yaml.safe_load(open(spec["base_config"]))
    #add this to give own run name and not copy from the configs/base file. 
    base_cfg["run_name"] = spec["run_name"]
    sweep_params = spec["sweep"]
    method       = spec.get("method", "grid").lower()
    sweep_id = spec_path.stem
    #replacement for "time based" run saving where each run of a sweep would have a different time. 
    sweep_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # one stamp per sweep


    if method == "grid":
        variants = list(grid_variants(sweep_params))
    elif method == "random":
        n = spec["random"]["num_samples"]
        variants = list(random_variants(sweep_params, n))
    else:
        raise ValueError(f"Unsupported sweep method '{method}'")

    # Handle optional seed fan‑out
    seeds = spec.get("seeds", [None])
     # (optional) bootstrap fan‑out
    boot_cfg = spec.get("bootstrap", {})
    do_boot  = boot_cfg.get("enabled", False)
    n_boot   = boot_cfg.get("num_bootstrap", 0)
    boot_base = boot_cfg.get("resample_seed_base", 0)
    tmp_dir = Path(".sweep_tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    manifest_rows = []
    print(f"BASE {base_cfg['run_name']}")
    for idx, overrides in enumerate(variants):
        for seed in seeds:
            cfg_variant  = apply_overrides(deepcopy(base_cfg), overrides)
            print(f"variant {cfg_variant['run_name']}")
            if seed is not None:
                cfg_variant["seed"] = seed
            # embed sweep metadata
            cfg_variant["_sweep"] = {"id": sweep_id, "idx": idx, "seed": seed, "stamp": sweep_stamp}
            # ------------------------------------------------------------------
            # Handle bootstrap fan‑out
            # ------------------------------------------------------------------
            boot_iter = range(n_boot) if do_boot else [None]
            for b in boot_iter:
                cfg_run = deepcopy(cfg_variant)
                if "run_name" not in cfg_run:
                    print("new run name: ")
                    cfg_run["run_name"] = sweep_id        # e.g. "extraGoodRun"
                print(cfg_run["run_name"])
                if b is not None:
                    cfg_run["bootstrap"] = {"enabled": True, "seed": boot_base + b}
                    cfg_run["_sweep"]["boot"] = b

                # write temp config
                #mode = "w" write to text stream rather than binary file mode. 
                with tempfile.NamedTemporaryFile(mode = "w", dir=tmp_dir,
                                                 suffix=".yaml",
                                                 delete=False) as tmp_f:
                    yaml.safe_dump(cfg_run, tmp_f)
                    tmp_path = Path(tmp_f.name)

                # launch training run
                launch_main(tmp_path)

                # record manifest entry
                m_entry = {
                    **cfg_run["_sweep"],
                    **overrides,               # hyper‑params
                    "temp_cfg": str(tmp_path)  # path for traceability
                }
                manifest_rows.append(m_entry)
    # ------------------------------------------------------------------
    # Save sweep manifest
    # ------------------------------------------------------------------
    #TODO: Change where sweep manifest is saved, do whatever metrics or things I want afterward. 
    man_path = Path("runs") / f"{sweep_id}_manifest.csv"
    man_path.parent.mkdir(exist_ok=True, parents=True)
    if manifest_rows:
        columns = sorted(set().union(*manifest_rows))
        with man_path.open("w", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=columns)
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow(row)

    print(f"[sweeper] Finished. Manifest saved to {man_path}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to sweep YAML.")
args = parser.parse_args()
launch_sweep(Path(args.config))
from pathlib import Path
import datetime
import yaml
import shutil
import json
import os
import platform
import subprocess
class RunManager:
    def __init__(self, config, base_dir="runs", resume=False, utility = False):
        self.utility = utility
        self.base_dir = Path(base_dir)
        if not utility:
            self.config = config
            self.resume = resume
            #could rewrite this to be more class based, but prefer the functional way. 
            self.run_dir, self.run_name = self._get_run_dir(naming = config.get("run_naming", "index"))
            #Need to add this here now. 
            config["run_name"] = self.run_name
            #not sure if pass the config or use self.config. Never sure. 
            self._save_config(config)
            #TODO: Add way to save metadata at the end of training. 
            self._save_metadata()
            
    @classmethod
    def utility_mode(cls, base_dir = "runs"):
        return cls(config = None, base_dir = base_dir, utility = True)   
    
    def _get_run_dir(self, naming):
        #TODO: Make sure resuming works, checks for the right thing. Don't know if need to add to config file. 
        name = self.config["run_name"]
        if self.resume: 
            #note: HEre the run_name SHOULD include the suffix. 
            run_dir = self.base_dir / name
            if not run_dir.exists():
                raise FileNotFoundError(f"No run found at {run_dir}")
            return run_dir, name
        else:
            run_dir, run_name = get_run_dir(self.base_dir, name, naming)
            return run_dir, run_name

    def _save_config(self, config):
        """
        Save config file in runs/config_name as both yaml (editable) and json (wandb). 
        
        Bonus: Check what to do for overwriting. 
        """
        #path where the runs are saved. Folder runs/runName/ stuff here
        config_path = self.run_dir / "config.yaml"
        if config_path.exists():
            #print(f"⚠️ Config already exists at {config_path}, overwriting.")
            raise FileExistsError(f"Run config already exists at {config_path}")
        with open(config_path, "w") as f:
            #could write something checking for overwriting/collision. 
            yaml.dump(config, f)
        #don't need to check json, config should already exist. But Potential source of error here. 
        with open(self.run_dir/"config.json", "w") as f:
            json.dump(config, f, indent = 2)

    
    def _save_metadata(self, status="started", locked=False):
        metadata = {
            "run_name": self.run_name,
            "start_time": datetime.datetime.now().isoformat(),
            "user": os.getenv("USER", "unknown"),
            "hostname": platform.node(),
            "git_commit": self._get_git_commit(),
            "git_dirty": self._is_git_dirty(),
            "python_version": platform.python_version(),
            "status": status,
            "locked": locked,
        }

        with open(self.run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_git_commit(self):
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except Exception:
            return "unknown"

    def _is_git_dirty(self):
        try:
            out = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
            return bool(out)
        except Exception:
            return False
    def list_runsL(self, experiment=None):
        if experiment:
            exp_dir = self.base_dir/ experiment
            if not exp_dir.exists():
                print(f"No experiment named '{experiment}' found.")
                return
            print(f"[{experiment}]")
            for sub in sorted(exp_dir.glob(f"{experiment}*")):
                if sub.is_dir():
                    print("  -", sub.name)
        else:
            for exp_folder in sorted(self.base_dir.glob("*")):
                if exp_folder.is_dir():
                    print(f"[{exp_folder.name}]")
                    for sub in sorted(exp_folder.glob(f"{exp_folder.name}*")):
                        print("  -", sub.name)
    
    def list_runs(self, base_name_filter=None):
        print("Available runs:\n")

        for run_group in sorted(self.base_dir.iterdir()):
            if not run_group.is_dir():
                continue
            if base_name_filter and base_name_filter not in run_group.name:
                continue

            for run_dir in sorted(run_group.iterdir()):
                if not run_dir.is_dir():
                    continue

                meta_path = run_dir / "run_metadata.json"
                status = "unknown"
                locked = False
                start_time = "?"

                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        status = meta.get("status", "unknown")
                        start_time = meta.get("start_time", "?")
                        locked = meta.get("locked", False)

                lock_str = "🔒" if locked else ""
                print(f"[{run_dir.name}] {lock_str} status: {status}, started: {start_time}")
        return 

    def delete_run(self, run_name):
        #TODO: Add the keep or lock mechanism. 
        run_dir = self.base_dir / run_name
        if not run_dir.exists():
            print(f"Run {run_name} does not exist.")
            return
        #need to deal with this. 
        if (run_dir / ".keep").exists():
            print(f"Run {run_name} is locked. Skipping deletion.")
            return
        shutil.rmtree(run_dir)
        print(f"Deleted run {run_name}")

    #allows for not needing to manually rename everything. 
def get_run_dir( base_dir, base_name, naming = "index"):
    """
    TODO: Fix sweeping for the index run naming scheme. 
    """
    base = Path(base_dir)/base_name #right now this is runs + name we chose. 
    base.mkdir(parents=True, exist_ok=True) #make the folder. 

    if naming == "timestamp": 
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{base_name}_{time_str}"
        
    elif naming == "index":
        #need to check this works. 
        existing = sorted([p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(base_name)])
        indices = [int(name.replace(base_name, '')) for name in existing if name.replace(base_name, '').isdigit()]
        next_idx = (max(indices) + 1) if indices else 0
        run_name = f"{base_name}{next_idx}"
    elif naming == "uuid":
        import uuid
        run_name = f"{base_name}_{uuid.uuid4().hex[:8]}"
    else: 
        raise ValueError(f"Unknown naming scheme: {naming}")
    run_path = base / run_name
    #make the run directory in the larger folder of that family of runs. 
    run_path.mkdir(parents = True, exist_ok = False)
    return run_path, run_name
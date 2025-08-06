from hooks.registry import get_registered_hook, get_hooks
from diagnostics.registry import get_diagnostic
import matplotlib
import time
#TODO: Fix the triggers/timing on everything. WHAT TO DO IF NOT IN CONFIG. 
def register_hooks_from_config(hook_mgr, config):
    #TODO: This line doesn't work. Are we sure? 
    for hook_cfg in config.get("hooks", get_hooks()):
        name = hook_cfg.get("name", None)
        reg = get_registered_hook(name)
        if not reg:
            print(f"[Hooks] Warning: hook '{name}' not found in registry.")
            continue

        fn = reg["fn"]
        trigger = hook_cfg.get("trigger", reg["trigger"])
        every   = hook_cfg.get("every", reg["every"])
        
        hook_mgr.register(fn, trigger=trigger, every=every, name=name)

def register_diagnostics_as_hooks(hook_mgr, config):
    #TODO: Fix this make sure it works. Problem with some diagnostic in what i'm doing. 
    diagnostics_to_run = config.get("diagnostics", [])

    for entry in diagnostics_to_run:
        if isinstance(entry, str):
            name = entry
            override = {}
        else:
            name = entry["name"]
            override = entry

        fn = get_diagnostic(name)
        if not fn:
            print(f"[Warning] Diagnostic '{name}' not found.")
            continue

        meta = getattr(fn, "_diagnostic_meta", {})
        trigger = override.get("trigger", meta.get("trigger", "epoch"))
        every   = override.get("every", meta.get("every", 5))

        hook_mgr.register(
            callback=make_hook_wrapper(fn, name, trigger),
            trigger=trigger,
            every=every,
            name=name
        )
def make_hook_wrapper(fn, name, trigger):
    def hook_wrapper(**kwargs):
        try:
            t0 = time.time()
            outputs = fn(**kwargs) or {}
            logger = kwargs["logger"]
            step = kwargs["step"]
            for k, v in outputs.items():
                """
                #was going to return figs and artifacts, instead will do in each method. 
                print("K: ", k)
                print("V: ", v)
                if v is matplotlib.figure.Figure():
                    #note: changed this to step not epoch to fix errors with wandb. 
                    logger.save_plot(v, k + ".png", step)
                elif v is 
                """
                #changed step_type to step, since we only want to log wandb stuff as steps. This means my logs will be per step as well. 
                logger.log_scalar(f"{name}/{k}", v, step=step, step_type="step")
            print(f"[Diagnostics] {name} finished in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"[Diagnostic-Hook] {name} failed: {e}")
    return hook_wrapper
"""
 def run_diagnostics(self, epoch):
        #runs the diagnostic functions in the diagnostic registry, allows us to not have to specify them here. 
        #specified in config file. 
        diagnostics_to_run = self.config.get("diagnostics", [])
        registry = get_diagnostics()
        #in the diagnostic function, log via log_scalar. 
        for name in diagnostics_to_run:
            fn = registry.get(name)
            if fn is None:
                print(f"[Diagnostics] Warning: diagnostic '{name}' not found in registry.")
                continue

            try:
                print(f"[Diagnostics] {name}")
                t0 = time.time()
                
                outputs = fn(self.model, self.val_loader, self.logger, epoch, self.config, self.meta) or {}
                #log diagnostic scalars here instead. 
                for field, value in outputs.items():
                    self.logger.log_scalar(f"{name}/{field}", value, epoch)
                print(f"[Diagnostics] {name} finished in {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"[Diagnostics] {name} failed: {e}")
"""
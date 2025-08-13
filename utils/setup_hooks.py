from diagnostics.registry import get_diagnostic, get_diagnostics
import matplotlib
import time
def register_hooks_from_config(hook_mgr, config):
    for hook_cfg in config.get("hooks", get_diagnostics()):
        name = hook_cfg.get("name", None)
        reg = get_diagnostic(name)
        if not reg:
            print(f"[Hooks] Warning: hook '{name}' not found in registry.")
            continue

        fn = reg["fn"]
        trigger = hook_cfg.get("trigger", reg["trigger"])
        every   = hook_cfg.get("every", reg["every"])
        
        hook_mgr.register(fn, trigger=trigger, every=every, name=name)


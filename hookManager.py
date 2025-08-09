class HookManager:
    
    def __init__(self):
        self.hooks = {"epoch": [], "step": [], "begin":[], "end":[]}

    def register(self, callback, trigger="epoch", every=1, condition=None, name = None):
        hook = Hook(name, callback, trigger, every, condition)
        self.hooks[trigger].append(hook)

    def call(self, trigger_point, trigger="epoch", **kwargs):
        for hook in self.hooks[trigger]:
            if hook.should_run(trigger_point):
                hook.callback(**kwargs)
    def list_hooks(self, trigger):
        for hook in self.hooks[trigger]:
            print(f"Hook: {hook.get_name()}\n")
class Hook:
    def __init__(self, name, callback, trigger="epoch", every=1, condition=None):
        self.name = name
        self.callback = callback
        self.trigger = trigger  # "epoch" or "step"
        self.every = every
        self.condition = condition  # Optional function(step, ...) -> bool

    def should_run(self, trigger_point):
        if self.condition:
            return self.condition(trigger_point)
        return trigger_point % self.every == 0
    def get_name(self):
        return self.name
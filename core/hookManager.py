from enum import Enum, auto
from dataclasses import dataclass, replace
from typing import Any, Callable
from utils.hookHelpers import * 

class HookManager:
    def __init__(self):
        self._hooks: dict[Trigger, list[Hook]] = {t: [] for t in Trigger}
    def register(self, name, callback, trigger:Trigger, every=1, condition=None, priority = 0):
        hook = Hook(name, callback, trigger, every, condition, priority)
        self._hooks[trigger].append(hook)
        self._hooks[trigger].sort(key=lambda h: h.priority) # Is this necessary? 

    def call(self, trigger: Trigger, ctx: StepCtx, services: Services):
        #TODO: See what adding hook output does. 
        results = []
        for hook in self._hooks[trigger]:
            if hook.should_run(ctx):
                out = hook.callback(ctx, services)    
                if out: results.append((hook.name, out)) 
        return results
    
    def list_hooks(self, trigger):
        for hook in self.hooks[trigger]:
            print(f"Hook: {hook.get_name()}\n")
@dataclass
class Hook:
    name: str
    callback: Callable
    trigger: Trigger
    every: int = 1
    condition: Callable[[StepCtx, Services], bool] | None = None
    priority: int = 0  # lower runs first

    def should_run(self, ctx: StepCtx):
        #TODO: Understand this right here! See the ramifications. 
        if self.condition and not self.condition(ctx, None): # if there is a condition method and calling the condition method returns false, don't run. 
            return False
        #Use epoch for epoch-level, step for step-level, simple rule. Did i change this? 
        idx = ctx.step if ctx.phase == "train" and ctx.batch_idx >=0 else ctx.epoch # what phases are there? 
        return (idx% max(self.every, 1)) == 0
    def get_name(self):
        return self.name

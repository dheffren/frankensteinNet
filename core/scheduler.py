from typing import Optional, Dict
###Generic hyperparameter scheduler. 
class ScalarSchedule:
    def __init__(self, schedule_type, initial, final, start_epoch, end_epoch):
        self.schedule_type = schedule_type
        self.initial = initial
        self.final = final
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def get(self, epoch):
        if epoch < self.start_epoch:
            return self.initial
        elif epoch >= self.end_epoch:
            return self.final

        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.schedule_type == "linear":
            return self.initial + progress * (self.final - self.initial)
        elif self.schedule_type == "exp":
            return self.initial * (self.final / self.initial) ** progress
        elif self.schedule_type == "constant":
            return self.initial
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")
class SchedBundle:
    """
    Central place to keep N ScalarSchedule objects.
    Call .get(name, epoch) or get_all(epoch) to retrieve values.
    """

    def __init__(self, cfg_block: Optional[Dict] = None, default_type="constant"):
        cfg_block  = cfg_block or {}
        self._sched = {
            name: ScalarSchedule(
                schedule_type=spec.get("type", default_type),
                initial=spec["initial"],
                final=spec.get("final", spec["initial"]),
                start_epoch=spec.get("start_epoch", 0),
                end_epoch=spec.get("end_epoch", 1),
            )
            for name, spec in cfg_block.items()
        }

    def get(self, name: str, epoch: int, default: float = 1.0):
        if name in self._sched:
            return self._sched[name].get(epoch)
        return default

    def get_all(self, epoch: int, defaults: dict[str, float] = None):
        out = {k: s.get(epoch) for k, s in self._sched.items()}
        #sets the defaults. 
        if defaults: 
            for k, v in defaults.items():
                out.setdefault(k,v)
        return out
    
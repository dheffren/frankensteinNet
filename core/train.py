from core.logger import Logger
import torch
import contextlib
from  diagnostics.registry import get_diagnostics
import time
from dataclasses import dataclass
from core.hookManager import HookManager, Trigger, StepCtx, Services
from collections import defaultdict
from utils.hookHelpers import Services
class AvgMeter:
    def __init__(self): self.s=0.0; self.n=0
    def update(self,x,n=1): self.s+=float(x); self.n+=n
    def mean(self): return self.s/max(self.n,1)

class MetricStore:
    def __init__(self): self.m={}
    def update(self, scalars: dict, prefix=""):
        for k,v in scalars.items():
            if isinstance(v, dict): self.update(v, prefix+f"{k}/"); continue
            key = prefix+k
            self.m.setdefault(key, AvgMeter()).update(v)
    def as_dict(self): return {k: m.mean() for k,m in self.m.items()}
    def reset(self): self.m.clear()
class MeterRegistry:
    #
    def __init__(self): self._m = {}
    def get(self, key):
        if key not in self._m: self._m[key] = MetricStore()
        return self._m[key]
    def reset(self, key): 
        if key in self._m: self._m[key].reset()
@dataclass(frozen=True)
class ACtx:  # minimal ArtifactContext shape
    run_id: str; epoch: int; step: int; trigger: str; hook: str; split: str|None=None
class Trainer:
    """
    Trainer Class - orchestrate training
    Builds Model, loads data, handles epochs and batches, tracks loos/accuracy. 
    Calls logger, optionally triggers diagnostics.

    Specify model, optimizer, dataset and diagnostics outside of the trainer class.  
    """
    def __init__(self, model, optimizer, scheduler, dataloaders, logger, hook_manager,meta, config):
        self.model = model
        self.train_loader, self.val_loader = dataloaders["train"], dataloaders["val"]
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meta = meta
        self.global_step = 0
        self.config = config
        self.hook_manager = hook_manager
        self.epoch = 0
        
        self.services = Services(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            logger=self.logger,
            cfg=self.config,
            meta=self.meta,
            device=self.config["device"], 
            run_eval=self._run_eval,
            checkpoint=self._checkpoint, 
            run_dir = self.logger.run_dir
        )
        
        self.meters = MeterRegistry()
        return 
    def _checkpoint(self, tag: str) -> str:
        #TODO: Fix this. 
        import io, os
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        from dataclasses import dataclass
        
        actx = ACtx(run_id=self.meta.get("run_id","run"), epoch=self.epoch, step=self.global_step,
                    trigger="checkpoint", hook="trainer", split=None)
        key = f"checkpoint/{tag}.pth"
        self.logger.save_bytes(actx, key, buf.getvalue())
        return key
    
    def _run_eval(self, split="val") -> dict:
            return self.evaluate(self.epoch, phase=split)
    # helper
    def _ctx(self, *, phase, trigger: Trigger, loss=None, metrics=None, batch_idx=-1):
        return StepCtx(
            phase=phase,
            epoch=self.epoch,
            step=self.global_step,
            batch_idx=batch_idx,
            trigger=trigger.name.lower(),
            hook=None,
            lr=self._lr_value(),
            loss=(float(loss) if loss is not None else None),
            metrics=metrics or {}
        )
    def _lr_value(self):
        groups = self.optimizer.param_groups
        if len(groups)==1: return groups[0]["lr"]
        return {i:g["lr"] for i,g in enumerate(groups)}
    def _update_meter(self, phase: str, scalars: dict):
        
        self.meters.get(phase).update(scalars)
        
    def _fire(self, trig: Trigger, ctx: StepCtx, finalize=False):
        """
        This method assumes returning stuff from the hooks methods. 
        """
        for name, out in self.hook_manager.call(trig, ctx, self.services) or []:
            if not out: continue
            if "metrics" in out and out["metrics"]:
                print("out metrics: ", out["metrics"])
                self.logger.log_dict(ctx, out["metrics"], finalize=False)
                # accumulate for epoch means on train steps only
                if ctx.phase=="train" and ctx.batch_idx>=0:
                    self._update_meter(ctx.phase, out["metrics"])
            # artifacts: your logger queues + uploads at step finalize
            if "artifacts" in out:
                #why do we need this exactly:? Should it be saved in the moment? Are we sure all of these things are equal? 
                # Expect artifacts like [("bytes",  key, data_bytes), ("figure",  key, fig)]
                for dict in out["artifacts"]:
                    typ, key, payload, split = dict.values()
                    actx = ACtx(run_id=self.meta.get("run_id","run"), epoch=ctx.epoch, step=ctx.step,
                    trigger=trig, hook=name, split=split)
                    print("HERE")
                    print("type: ", typ)
                    print("Lkey: ", key)
                    print("Payload: ", payload)
                    print("split: ", split)
                    if typ=="bytes":
                        print("CALLING ART")
                        self.logger.save_artifact(actx, key, payload)
                    elif typ=="figure":
                        print("CALLING PLOT")
                        self.logger.save_plot(actx, key=key, fig=payload)
        if finalize:
            #Call each step and at end of epoch. 
            assert(ctx.step == self.global_step)
            self.logger.flush(ctx.step)
    def train(self):
        #TODO: log_dict currently setup to work with the train: epoch_means but i don't actually think that's the best way. I think it will make things worse. 
        try: 
            print("cudnn benchmark is enabled:", torch.backends.cudnn.benchmark) 
            torch.backends.cudnn.benchmark = True
            #metrics at begin of training. 
            ctx = self._ctx(phase = "train", trigger = Trigger.TRAIN_BEGIN)
            self._fire(Trigger.TRAIN_BEGIN, ctx)
            for self.epoch in range(self.config["training"]["epochs"]):
                self.train_epoch(self.epoch)
                # EPOCH_END (train epoch means)
                epoch_means = self.meters.get("train").as_dict()
                ctx_end = self._ctx(phase="train", trigger=Trigger.EPOCH_END, metrics=epoch_means)
                self.logger.log_dict(ctx_end, epoch_means, finalize=False)
                self.meters.reset("train")
                self._fire(Trigger.EPOCH_END, ctx_end, finalize = True)
                return_dict_val = self.evaluate(self.epoch, True)
             
                print(f"Epoch {self.epoch}: Train {epoch_means['loss']}, Val {return_dict_val['loss']}")
                #update scheduler - if no scheduler, should still work as a constant. 
                self.scheduler.step() #-- if want to update lr in the middle of epoch, will have to do in train epoch. 
                #if i want per epoch values for train metrics somewhere, I have to DO the computation - it won't just give me everything. 
                # log things we care about. 
                
            # TRAIN_END
            ctx = self._ctx(phase="train", trigger=Trigger.TRAIN_END)
            self._fire(Trigger.TRAIN_END, ctx, finalize=True)
        except Exception as e:
            print("EXCEPTION")
            # EXCEPTION (ensure weights restored, allow hooks to dump state)
            ctx = self._ctx(phase="train", trigger=Trigger.EXCEPTION)
            self._fire(Trigger.EXCEPTION, ctx, finalize=True)
            raise 
        
    def train_epoch(self, epoch):
        self.model.train()
        ctx_epoch_begin = self._ctx(phase = "train", trigger = Trigger.EPOCH_BEGIN)
        self._fire(Trigger.EPOCH_BEGIN, ctx_epoch_begin)
        for (bidx, batch) in enumerate(self.train_loader):
            ctx_before = self._ctx(phase="train", trigger=Trigger.BEFORE_STEP, batch_idx=bidx)
            self._fire(Trigger.BEFORE_STEP, ctx_before)
            self.optimizer.zero_grad()
            #should only contain primary loss and any loss components or fast metrics. 
            t0 = time.time()
            loss_dict = self.model.compute_loss(batch, epoch)
            fwd_time = time.time() - t0
            #TODO: Check if this stuff works and is useful. 
            step_metrics = {**loss_dict, "time/forward": fwd_time}
            loss = loss_dict["loss"]
            ctx_af = self._ctx(phase="train", trigger=Trigger.AFTER_FORWARD, loss=loss, metrics=step_metrics, batch_idx=bidx)
            # log cheap scalars now (not finalized)
            self.logger.log_dict(ctx_af, step_metrics, finalize=False)
            self._update_meter("train", step_metrics)
            self._fire(Trigger.AFTER_FORWARD, ctx_af)
            #Backward
            t1 = time.time()
            
            loss.backward()
            bwd_time = time.time() - t1
            ctx_ab = self._ctx(phase="train", trigger=Trigger.AFTER_BACKWARD,
                               loss=loss, metrics={"time/backward": bwd_time}, batch_idx=bidx)
            self.logger.log_dict(ctx_ab,  {"time/backward": bwd_time}, finalize=False)
            self._update_meter("train",  {"time/backward": bwd_time})
            self._fire(Trigger.AFTER_BACKWARD, ctx_ab)
            
            # ---- BEFORE OPT STEP
            ctx_pre = self._ctx(phase="train", trigger=Trigger.BEFORE_OPT_STEP, loss=loss, batch_idx=bidx)
            self._fire(Trigger.BEFORE_OPT_STEP, ctx_pre)

            self.optimizer.step()
            
            # ---- AFTER OPT STEP (good place to finalize step)
            ctx_post = self._ctx(phase="train", trigger=Trigger.AFTER_OPT_STEP, loss=loss, batch_idx=bidx)
            # Let hooks add weight norms, EMA, etc.; finalize this step once.
            self._fire(Trigger.AFTER_OPT_STEP, ctx_post, finalize=True)
            self.global_step += 1
        return

    def evaluate(self, epoch, use_gradients = False, step_log= False, phase = "val"):
        #maybe only include jacobian terms in training not validation. 
        self.model.eval()
        
        ctx_eval_begin = self._ctx(phase=phase, trigger=Trigger.EVAL_BEGIN)
        self._fire(Trigger.EVAL_BEGIN, ctx_eval_begin)
        t_total = 0
        step_count = 0
        #if in case we need to compute some kind of gradient. I don't love this though. 
        if use_gradients: 
            context = contextlib.nullcontext()
        else:
            context = torch.no_grad()
            print("No grad")
        with context:
            for (bidx, batch) in enumerate(self.val_loader if phase=="val" else self.train_loader): 
                t0 = time.time()
                loss_dict = self.model.compute_loss(batch, epoch)
                t_total +=time.time()-t0
                step_count+=1

                # accumulate like train, but don't advance global_step
                per_batch = {k: float(v.item()) for k, v in loss_dict.items()} #DO I NEED TO DO THIS? 
                
                self._update_meter(phase, per_batch)
                # Optional: per-batch eval hooks (off by default)
                # ctx_step = self._ctx(phase=phase, trigger=Trigger.EVAL_STEP, batch_idx=bidx, metrics=per_batch)
                # self._fire(Trigger.EVAL_STEP, ctx_step)

        avg_loss= self.meters.get(phase).as_dict()
        avg_time = {"time/batch": t_total/step_count}
        out = {**avg_loss, **avg_time}
        # EVAL_END (log val means; do not bump global_step)
        ctx_eval_end = self._ctx(phase=phase, trigger=Trigger.EVAL_END, metrics=out)
        self.logger.log_dict(ctx_eval_end, out, finalize = False) # Should finalize be true? 
        self._fire(Trigger.EVAL_END, ctx_eval_end, finalize = True)
        self.meters.reset(phase)
        return out
    
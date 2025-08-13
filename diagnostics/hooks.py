from .registry import register_diagnostic
import torch
#TODO: Make it so can use same method for epoch and step? OR waste of time? Would need to rework hooks as well. 
@register_diagnostic("epoch",default_trigger = "epoch", default_every = 1)
def log_epoch(name, trigger, step,  logger, epoch,  **kwargs):
       # print("logging ep")
        logger.log_scalar(f"{name}", epoch, step=step)
@register_diagnostic("train_metrics", default_trigger = "epoch", default_every = 1)
def log_train_metrics(name, trigger, step, logger,  train_metrics,  **kwargs):
  
    for namee, dict in train_metrics.items():
        for k,v in dict.items():
            if torch.is_tensor(v):
                    v = v.item()
            logger.log_scalar(f"{name}/{trigger}/{namee}/{k}", v, step)
@register_diagnostic("val_metrics", default_trigger = "epoch", default_every = 1)
def log_val_metrics_epoch(name, trigger, step, logger, val_metrics, **kwargs):
    for namee, dict in val_metrics.items():
        for k, v in dict.items():
            if torch.is_tensor(v):
                    v = v.item()
            #print(f"val/{name}/{k}")
            logger.log_scalar(f"{name}/{trigger}/{namee}/{k}", v, step)
@register_diagnostic("checkpoints", default_trigger = "epoch", default_every = 10)
def save_checkpoints(name, trigger, step, model,   logger, epoch, **kwargs):
    #print("saving check")
    logger.save_checkpoint(model, epoch) #this epoch isn't a problem.
@register_diagnostic("learning_rate", default_trigger = "epoch", default_every = 1)
def log_learning_rate(name, trigger, step, logger, lr,**kwargs):
    logger.log_scalar("lr", lr, step)





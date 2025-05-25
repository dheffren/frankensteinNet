from .registry import register_hook
#TODO: Make it so can use same method for epoch and step? OR waste of time? Would need to rework hooks as well. 
@register_hook("log_epoch",default_trigger = "epoch", default_every = 1)
def log_epoch(step,  logger, epoch, step_type, **kwargs):
        print("logging ep")
        logger.log_scalar("epoch", epoch, step=step, step_type=step_type)
@register_hook("log_train_metrics", default_trigger = "epoch", default_every = 1)
def log_train_metrics_epoch(step, logger,  train_metrics, step_type, **kwargs):
    print("logging met")
    for name, dict in train_metrics.items():
        for k,v in dict.items():
            logger.log_scalar(f"train/{name}/{k}", v, step, step_type = step_type)
@register_hook("log_val_metrics", default_trigger = "epoch", default_every = 1)
def log_val_metrics_epoch(step, logger, val_metrics, step_type, **kwargs):
    print("logging val")
    for name, dict in val_metrics.items():
        for k, v in dict.items():
            print(f"val/{name}/{k}")
            logger.log_scalar(f"val/{name}/{k}", v, step, step_type = step_type)
@register_hook("log_checkpoints", default_trigger = "epoch", default_every = 10)
def save_checkpoints(step, model,   logger, epoch, **kwargs):
    print("saving check")
    logger.save_checkpoint(model, epoch) #this epoch isn't a problem.
@register_hook("log_learning_rate", default_trigger = "epoch", default_every = 1)
def log_learning_rate(step, logger, lr, step_type, **kwargs):
    logger.log_scalar("lr", lr, step, step_type = step_type)
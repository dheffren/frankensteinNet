from logger import Logger
import torch
import contextlib
from  diagnostics.registry import get_diagnostics
import time
class Trainer:
    """
    Trainer Class - orchestrate training
    Builds Model, loads data, handles epochs and batches, tracks loos/accuracy. 
    Calls logger, optionally triggers diagnostics.

    Specify model, optimizer, dataset and diagnostics outside of the trainer class.  
    """
    def __init__(self, model, optimizer, scheduler, dataloaders, logger, config):
        self.model = model
        self.train_loader, self.val_loader = dataloaders
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0
        self.config = config
        self.device = config["device"]
        return 
    def train(self):
        for epoch in range(self.config["training"]["epochs"]):
            train_loss = self.train_epoch(epoch)

            val_loss = self.evaluate()
            print(f"Epoch {epoch}: Train loss{train_loss}, Val{val_loss}")
            #update scheduler - if no scheduler, should still work as a constant. 
            self.scheduler.step() #-- if want to update lr in the middle of epoch, will have to do in train epoch. 
            # log things we care about. 
            #TODO: Maybe make a separate one for each of validation and train loss. 
            if epoch % self.config["training"]["log_interval"] ==0:
                self.logger.log_scalar("train/loss", train_loss, epoch)
                self.logger.log_scalar("val/loss", val_loss, epoch)
            #log the current learning rate: OPTIONAL. 
            if self.config["scheduler"]["log"]:
                self.logger.log_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            #save checkpoints here. And log anything else we want. 
            #TODO: Add some method "of checking" whether to save checkpoints or not. Maybe add parameter to the config. 
            if epoch % self.config["training"]["diagnostic_interval"] == 0:
                if self.config["training"]["save_checkpoints"]:
                    self.logger.save_checkpoint(self.model, epoch)
                # log model state or latent space here. 
                self.run_diagnostics(epoch) #with grad"
            #logging per epoch - done manually here. 
            self.logger.flush(epoch)
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            #should only contain primary loss and any loss components or fast metrics. 
            loss_dict = self.model.compute_loss(batch)
            #TODO: Add scheduler (custom hyperparameter schedulers) here and add to the config. 
            loss = loss_dict["loss"]
            loss.backward()
            self.optimizer.step()
            #why item? 
            total_loss+=loss.item()
            #per batch logging - much more volatile. 
            #TODO: Potentially check which things we want to log here via the config file. 
            #THIS HERE DOESN'T WORK WITH THE CSV, MAKES IT LOOK UGLY. 
            #for k,v in loss_dict.items():
                 #self.logger.log_scalar(f"train/{k}", v.item(), self.global_step)
            #what on earth is this? - total number of optimizer steps. 
            self.global_step += 1
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    def evaluate(self, use_gradients = False):
        #maybe only include jacobian terms in training not validation. 
        total_loss = 0
        self.model.eval()
        #if in case we need to compute some kind of gradient. 
        if use_gradients: 
            context = contextlib.nullcontext()
        else:
            context = torch.no_grad()

        with context:
            for batch in self.val_loader: 
                batch = self._to_device(batch)
                loss_dict = self.model.compute_loss(batch)
                total_loss +=loss_dict["loss"].item()
                #don't do per batch evaluation. 
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def run_diagnostics(self, epoch):
        #runs the diagnostic functions in the diagnostic registry, allows us to not have to specify them here. 
        #specified in config file. 
        diagnostics_to_run = self.config.get("diagnostics", [])
        registry = get_diagnostics()
        #in the diagnostic function, log via log_scalar. 
        for name in diagnostics_to_run:
            print("name: ", name)
            fn = registry.get(name)
            if fn is None:
                print(f"[Diagnostics] Warning: diagnostic '{name}' not found in registry.")
                continue

            try:
                print(f"[Diagnostics] {name}")
                t0 = time.time()
                #TODO: Setup diagnostic functions. 
                fn(self.model, self.val_loader, self.logger, epoch, self.config)
                print(f"[Diagnostics] {name} finished in {time.time() - t0:.2f}s")
            except Exception as e:
                print(f"[Diagnostics] {name} failed: {e}")

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [x.to(self.device) for x in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        else:
            return batch.to(self.device)
from logger import Logger
import torch
import contextlib
from  diagnostics.registry import get_diagnostics
import time
from collections import defaultdict
class Trainer:
    """
    Trainer Class - orchestrate training
    Builds Model, loads data, handles epochs and batches, tracks loos/accuracy. 
    Calls logger, optionally triggers diagnostics.

    Specify model, optimizer, dataset and diagnostics outside of the trainer class.  
    """
    def __init__(self, model, optimizer, scheduler, dataloaders, logger, meta, config):
        self.model = model
        self.train_loader, self.val_loader = dataloaders["train"], dataloaders["val"]
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.meta = meta
        self.global_step = 0
        self.config = config
        return 
    def train(self):
        #PLACEHOLDER
        self.model.train()
        print("cudnn benchmark is enabled:", torch.backends.cudnn.benchmark) 
        torch.backends.cudnn.benchmark = True
        for epoch in range(self.config["training"]["epochs"]):
            #TODO: Placeholder
            use_gradients = True
            return_dict_train = self.train_epoch(epoch)
            #log gradient norms here. 
            
            return_dict_val = self.evaluate(epoch, use_gradients)
            print(f"Epoch {epoch}: Train {return_dict_train['loss']['loss']}, Val {return_dict_val['loss']['loss']}")
            #update scheduler - if no scheduler, should still work as a constant. 
            self.scheduler.step() #-- if want to update lr in the middle of epoch, will have to do in train epoch. 
            # log things we care about. 
            #TODO: Maybe make a separate one for each of validation and train loss, and grad norm. Shouldn't this just be avg tho? 
            if epoch % self.config["training"]["log_interval"] ==0:
                for name, dict in return_dict_train.items():
                    for k,v in dict.items():
                        #print("v: ", v)
                        self.logger.log_scalar(f"train/{name}/{k}", v, epoch)
                for name, dict in return_dict_val.items():
                    for k, v in dict.items():
                        self.logger.log_scalar(f"val/{name}/{k}", v, epoch)
            #log the current learning rate: OPTIONAL. 
            if self.config["scheduler"]["log"]:
                self.logger.log_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            #save checkpoints here. And log anything else we want. 
            if epoch % self.config["training"]["diagnostic_interval"] == 0:
                if self.config["training"]["save_checkpoints"]:
                    self.logger.save_checkpoint(self.model, epoch)
                # log model state or latent space here. 
                self.run_diagnostics(epoch) #with grad"
            #logging per epoch - done manually here. 
            self.logger.flush(epoch)
        
    def train_epoch(self, epoch, step_log = False):
        self.model.train()
        avg_batch_time = 0
        dict_dict = {"loss": {}, "grad_norm":{}, "weight_norm":{}, "time": {}}
        val_dict = {"loss": {}, "grad_norm":{}, "weight_norm":{}, "time": {}}
        step_count = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            #should only contain primary loss and any loss components or fast metrics. 
            start = time.time()
            loss_dict = self.model.compute_loss(batch, epoch)
            
            
            loss = loss_dict["loss"]
            loss.backward()
                
            grad_norms = self.compute_gradient_norms(self.model, group_layers=True)
            weight_norms = self.compute_weight_norms(self.model, group_layers = True)
            val_dict["loss"] = loss_dict
            val_dict["grad_norm"] = grad_norms
            val_dict["weight_norm"] = weight_norms 
            self.optimizer.step()
            end = time.time()
            val_dict["time"] = {"time": end-start}
            dict_dict = self.track_vals(dict_dict, val_dict, step_log)
            #what on earth is this? - total number of optimizer steps. 
            self.global_step += 1
            step_count+=1
        #slower if not logging losses same way but whatever. 
        dict_dict = {name: {k: v/step_count for k,v in dict.items()} for name, dict in dict_dict.items()}
      
        return dict_dict
    def track_vals(self, dict_dict, val_dict, step_log = False):
        #val list should be same length as dict_dict? Or not. 
        for name, dict in val_dict.items():
            #now looking at a specific val dict
            for k, v in dict.items():
                if k not in dict_dict[name].keys():
                    dict_dict[name][k] = 0.0
                if name is "loss": 
                    v = v.item()
                dict_dict[name][k] += v
                if step_log: 
                    #maybe track metrics additionally. 
                    self.logger.log_scalar(f"train_step/{name}/{k}", v.item(), self.global_step)
        #TODO: Returning from here "rounds off" values in a way i don't like. 
        return dict_dict
    def evaluate(self, epoch, use_gradients = False, step_log= False):
        #maybe only include jacobian terms in training not validation. 
       
        avg_loss_dict = {}
        avg_batch_time = 0
        self.model.eval()
        step_count = 0
        #if in case we need to compute some kind of gradient. 
        if use_gradients: 
            context = contextlib.nullcontext()
        else:
            context = torch.no_grad()

        with context:
            for batch in self.val_loader: 
                start = time.time()
                loss_dict = self.model.compute_loss(batch, epoch)
                end = time.time()
                avg_batch_time += end-start
                for k,v in loss_dict.items():
                    if k not in avg_loss_dict.keys():
                        avg_loss_dict[k] = 0
                    avg_loss_dict[k] += v.item()
                    if step_log:
                        self.logger.log_scalar(f"val_step/{k}", v.item(), self.global_step)
                #should I do this? 
                self.global_step+=1
                step_count+=1
                #don't do per batch evaluation. 
        avg_loss= {k: v/step_count for k,v in avg_loss_dict.items()}
        avg_batch_time = {"time": avg_batch_time/step_count}
        return_dict ={"loss": avg_loss, "time": avg_batch_time}
        return return_dict
    def compute_gradient_norms(self, model, group_layers = False):
       
        """
        Computes gradient norms for all parameters in the model.

        Args:
            model (torch.nn.Module): The model with gradients computed.
            group_layers (bool): If True, aggregate gradients by layer prefix (e.g. 'encoder.0').

        Returns:
        #NOT TRUE, CHANGE THIS. 
            A dict containing:
                - per_param: {param_name: norm}
                - per_layer: {layer_name: norm} (if group_layers=True)
                - total: float (global gradient norm)
        """
        grad_norms = {}
        total_norm_sq = 0.0
        
        for name, param in model.named_parameters():
           
            if param.grad is not None:
                #detach grad from everything. Have gradient bc backprop. 
                norm = param.grad.detach().norm(2).item()
                grad_norms[name] = norm
                total_norm_sq += norm ** 2
        #hopefully no shared names
        grad_norms["total"] = total_norm_sq**0.5

        if group_layers:
            layer_norms = defaultdict(list)
            for name, norm in grad_norms.items():
                
                layer_name = name.split('.')[0]  # You can customize this grouping rule
           
                layer_norms[layer_name].append(norm ** 2)
            
            for layer, norm_sq_list in layer_norms.items():
                grad_norms[layer] = sum(norm_sq_list) ** 0.5
  
        return grad_norms
    def compute_weight_norms(self, model, group_layers = False):
        """
        Returns a dict of L2 norms of all model weights.
        """
        norms = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                norms[f"weight_norm/{name}"] = param.data.norm(2).item()
        #add per layer here as well. 
        if group_layers: 
            layer_norms = defaultdict(list)
            for name, norm in norms.items():
                layer_name = name.split('.')[0]
                layer_norms[layer_name].append(norm**2)
            for layer, norm_sq_list in layer_norms.items():
                norms[layer] = sum(norm_sq_list)**.5
        return norms
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

   
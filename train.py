from logger import Logger
import torch
import contextlib
from  diagnostics.registry import get_diagnostics
import time
from hookManager import HookManager
from collections import defaultdict

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

        return 
    

    def train(self):
        #PLACEHOLDER
        self.model.train()
        self.prerun_diagnostics()
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
            self.hook_manager.call(trigger_point = epoch, trigger = "epoch", 
                                   step = self.global_step, 
                                   model = self.model, 
                                   logger = self.logger, 
                                   val_loader = self.val_loader, 
                                   epoch = epoch, 
                                   cfg = self.config, 
                                   meta = self.meta, 
                                   train_metrics = return_dict_train, 
                                   val_metrics = return_dict_val, 
                                   step_type = "epoch", 
                                   lr = self.optimizer.param_groups[0]["lr"]
                                   )
            #logging per epoch - done manually here. 
            self.logger.flush(self.global_step)
        
    def train_epoch(self, epoch, step_log = False):
        self.model.train()

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
            dict_dict = self.track_vals(dict_dict, val_dict, epoch, step_log)
            #what on earth is this? - total number of optimizer steps. 
            self.global_step += 1
            step_count+=1
        #slower if not logging losses same way but whatever. 
        dict_dict = {name: {k: v/step_count for k,v in dict.items()} for name, dict in dict_dict.items()}
      
        return dict_dict
    def track_vals(self, dict_dict, val_dict, epoch, step_log = False):
        #val list should be same length as dict_dict? Or not. 
        for name, dict in val_dict.items():
            #now looking at a specific val dict
            for k, v in dict.items():
                if k not in dict_dict[name].keys():
                    dict_dict[name][k] = 0.0
                if name is "loss": 
                    v = v.item()
                dict_dict[name][k] += v
                   
        #TODO: Returning from here "rounds off" values in a way i don't like.
        # Extra stuff here - but i don't really have anything.  
        if step_log and hasattr(self, "hook_manager"):
            self.hook_manager.call(
                trigger_point = self.global_step,
                trigger="step",
                step=self.global_step,
                model=self.model,
                logger=self.logger,
                epoch=epoch,
                train_metrics=val_dict,
                step_type = "step", 
                config=self.config,
                meta=self.meta
            )
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
                if step_log and hasattr(self, "hook_manager"):
                    self.hook_manager.call(
                    trigger_point = self.global_step,
                    trigger="step",
                    step=self.global_step,
                    model=self.model,
                    logger=self.logger,
                    epoch=epoch,
                    val_metrics={"loss": loss_dict},
                    step_type = "step", 
                    config=self.config,
                    meta=self.meta
                )
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
   

    def prerun_diagnostics(self):
        ### Run certain diagnostic things at the beginning of the run, both for tracking and other needs. 
        #TODO: Make this more generalizable. 
        diag_cfg = self.config["diagnostics_config"]
        from diagnostics.helper import run_pca_analysis, compute_latent_all
        layers = diag_cfg.get("layer_pca_layers", ["latent"])
        max_batches = diag_cfg.get("max_batches", 120)#TODO: This may be a problem
        n_components = diag_cfg.get("layer_pca_components", 5)
        for layer in layers: 
            latents, labels = compute_latent_all(self.model, self.val_loader, layer, max_batches)
            projections, outputDict = run_pca_analysis(latents, labels, f"{layer}", self.logger, -1, n_components, None, None, False, self.meta)
            #if want to plot this stuff at the beginning of epoch. , can also do the prerun hooks. 
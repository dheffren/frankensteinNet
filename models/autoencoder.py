import torch
import torch.nn as nn

#TODO: Finish implementing at least one model. 
class Autoencoder(nn.Module):
    def __init__(self, model_cfg, loss_fn, hyp_sched,  metadata, device = "cpu", track_grad = True):
        super().__init__()
        
        #maybe should make this part modular somehow? So it doesn't have to be model specific? 
        input_dim = model_cfg.get("input_dim", 784)
        latent_dim = model_cfg.get("latent_dim", 16)
        hidden_dim = model_cfg.get("hidden_dim", 128)

        self.encoder = torch.nn.Sequential(
            
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
            )
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            #use sigmoid? 
            #nn.Sigmoid()
        )
        self.loss_fn = loss_fn
        self.hyp_sched = hyp_sched
        self.device = device
        
    def forward(self, x):
        #TODO: Add shape diagnostic. Not sure how lol. 
        x = x.view(x.shape[0], -1)
        latent = self.encoder(x)

        reconstruction = self.decoder(latent)
        #generalize this to deeper things. 
        #need to generalize this better
        reconstruction = reconstruction.view(x.shape[0], 1, 28, 28)
 
        return {"recon":{"x": reconstruction}, "latent":{"latent":latent}}
    def compute_loss(self, batch, epoch):
        #pass in forward method to support multiple tasks/losses. 
        #maybe instead of this do something else. 
        inputs, targets = self.prepare_input(batch)
        out = self(**inputs)
        #UPDATE THE SCHEDULED HYPERPARAMETERS HERE AND PASS INTO LOSS FUNCTION. Use epoch. 
        return self.loss_fn(out, targets)
    def prepare_input(self, batch, requires_grad = True):
        if isinstance(batch, torch.Tensor):
            #shouldn't we return x as an aux as well. 
            x = batch
            inputs  = {"x": x.to(self.device).requires_grad_(requires_grad)}
            targets = {"recon_target": {"x": x}}                    
            return inputs, targets
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 1:
                x = batch[0]
                inputs  = {"x": x.to(self.device).requires_grad_(requires_grad)}
                targets = {"recon_target": {"x": x}}                    
            elif len(batch) >= 2:
                #don't do anything with the rest - don't know what to do lol. 
                x, y = batch[:2]
                inputs  = {"x": x.to(self.device).requires_grad_(requires_grad)}
                targets = {"recon_target": {"x": x}, "labels":{"y": y}}  # recon + label
            return inputs, targets
        elif isinstance(batch, dict):
            #TODO: Make this more general? 
            #I dont' understand this code. 
            x = batch["x"]
            y = batch["y"]
            inputs  = {"x": x.to(self.device).requires_grad_(requires_grad)}
            #inputs  = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                    #   for k, v in batch.items()
                      # if k in {"x"} or k.startswith("x")}  # whatever you forward
            #ideally would make more generic, don't really have to here though. 
            targets = {"recon_target": {"x": x}, "labels":{"y": y}}  # recon + label
                      # recon target by default
            return inputs, targets

        raise TypeError("Unknown batch format for prepare_input")

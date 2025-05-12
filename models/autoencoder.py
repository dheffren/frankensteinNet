import torch
import torch.nn as nn
from losses import make_loss_fn
#TODO: Finish implementing at least one model. 
class Autoencoder(nn.Module):
    def __init__(self, model_cfg, loss_cfg):
        super().__init__()
        
        #maybe should make this part modular somehow? So it doesn't have to be model specific? 
        input_dim = model_cfg["input_dim"]
        latent_dim = model_cfg["latent_dim"]
        hidden_dim = model_cfg["hidden_dim"]

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
        self.loss_fn = make_loss_fn(loss_cfg)

    
    def forward(self, x):
        #do i have to say it is x? 
        #IS THIS REFERENCE OR ORIGINAL? SEE IN BACKUP LOSS. 
        #print("x shape: ", x.shape)
        #print(f"x: {x.shape}")
        x = x.view(x.shape[0], -1)
       # print(x.max())
        #print(x.min())
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        #print(f"Reconstruction shape: {reconstruction.shape}")
        #print(f"Reconstruction min/max: {reconstruction.min()}/{reconstruction.max()}")
        #generalize this to deeper things. 
        #need to generalize this better
        reconstruction = reconstruction.view(x.shape[0], 1, 28, 28)
        #print(f"Reconstruction shape2: {reconstruction.shape}")
        #print(f"Reconstruction min/max: {reconstruction.min()}/{reconstruction.max()}")
        return {"recon":reconstruction, "latent":latent}
    def compute_loss(self, batch):
        #pass in forward method to support multiple tasks/losses. 
        #maybe instead of this do something else. 
        return self.loss_fn(batch, self.forward)
    
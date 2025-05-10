import torch
import torch.nn as nn
#TODO: Finish implementing at least one model. 
class Autoencoder(nn.Module):
    def __init__(self, input_dim  = 784, latent_dim = 128, hidden_dim = 512, loss_fn= None):
        super().__init__()
        self.input_dim = input_dim
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
            nn.Sigmoid()
        )
        self.loss_fn = loss_fn
        if loss_fn is None:
            self.loss_fn = self.backup_loss
    
    def forward(self, x):
        #do i have to say it is x? 
        #IS THIS REFERENCE OR ORIGINAL? SEE IN BACKUP LOSS. 
        #print("x shape: ", x.shape)
        x = x.view(x.shape[0], -1)
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        #generalize this to deeper things. 
        #need to generalize this better
        reconstruction = reconstruction.view(x.shape[0], 1, 28, 28)
        return reconstruction, latent
    def compute_loss(self, batch):
        #pass in forward method to support multiple tasks/losses. 
        #maybe instead of this do something else. 
        return self.loss_fn(batch, self.forward)
    def backup_loss(self, batch, forward):
        x,y = batch
        reconstruction, latent = forward(x)
        print(reconstruction.shape)
        print(x.shape)
        l2loss = torch.mean(torch.sum((reconstruction - x)**2, axis=(-1, -2)))
        loss_dict = {"loss": l2loss}
        return loss_dict
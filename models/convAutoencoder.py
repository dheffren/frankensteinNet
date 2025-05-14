import torch
import torch.nn as nn
import math

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dimU= 2, latent_dimV =2, latent_dimC = 2):
        super(ConvolutionalAutoencoder, self).__init__()
        #assuming iamge dim 256x256
        in_channels = 1
        out_channels = 8
        kernel_size = 3
        #right padding size to counteract kernel. 
        pad = math.floor((kernel_size - 1)/2)
        self.convEncoder = ConvolutionalEncoder(in_channels, out_channels)
        
        self.convDecoder = ConvolutionalDecoder(in_channels, out_channels)
        #need to relate this to the number of layers somehow. 
        self.middleOrthog = False
        self.embed_dim = 4
        indim = 20
        self.project1 = nn.Sequential(
            nn.Linear(self.embed_dim * self.embed_dim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim),
            nn.Linear(indim, latent_dimU)
        )
        self.project2 = nn.Sequential(
            nn.Linear(self.embed_dim*self.embed_dim,  indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim),
            nn.Linear(indim, latent_dimV)
        )
        self.orthog1 = nn.Sequential(nn.Linear(latent_dimU, latent_dimU), nn.ReLU(), nn.Linear(latent_dimU, latent_dimU), nn.ReLU(), nn.Linear(latent_dimU, latent_dimU))
        self.orthog2 = nn.Sequential(nn.Linear(latent_dimV, latent_dimV), nn.ReLU(), nn.Linear(latent_dimV, latent_dimV), nn.ReLU(), nn.Linear(latent_dimV, latent_dimV))
        dedim = 40
        #this is RIDICULOUS
        self.deproject = nn.Sequential(
            nn.Linear(latent_dimU + latent_dimV, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim),
            nn.Linear(dedim,self.embed_dim*self.embed_dim)
        )
    
    def outputImage(self, u, v): 
        uv = torch.cat([u,v], dim=-1)
        uc = self.deproject(uv)
        x_recon = self.convDecoder(uc.view(-1, 1, self.embed_dim, self.embed_dim))
        return x_recon
    def outputu(self, x, v):
        uh = self.convEncoder(x)
        if self.middleOrthog:
            uh = self.project1(uh)
            u = self.orthog1(uh)
        else:
            u = self.project1(uh)

        uv = torch.cat([u,v], dim=-1)
        uc = self.deproject(uv)
        x_recon = self.convDecoder(uc.view((-1, 1, self.embed_dim, self.embed_dim)))
        return x_recon, u
    def outputv(self, x, u):
        uh = self.convEncoder(x)
        if self.middleOrthog:    
            uh = self.project1(uh)
            v = self.orthog2(uh)     
        else:
            v = self.project2(uh)
        
        uv = torch.cat([u,v], dim=-1)
        uc = self.deproject(uv)
        x_recon = self.convDecoder(uc.view((-1, 1, self.embed_dim, self.embed_dim)))
        return x_recon, v
    def forward(self, x):
        #NOTE: When Have those extra layers but take a gradient skipping over those, weird things happen. 
        uh = self.convEncoder(x)  # Encode S1 (Rotations)
    
    
    # u1 = self.project1(torch.flatten(u1h, 1))
        #u2 = self.project2(torch.flatten(u2h, 1))
        #ug = self.project1(uh)
        #u = self.orthog1(ug)
        if self.middleOrthog:
            uh = self.project1(uh)
            u = self.orthog1(uh)
            v = self.orthog2(uh)
        else:
            u = self.project1(uh)
            v = self.project2(uh)
    # vg = self.project2(uh)
    # v = self.orthog2(vg)
        uv = torch.cat([u,v], dim =-1)
        uc = self.deproject(uv)
        x_recon = self.convDecoder(uc.view((-1, 1, self.embed_dim, self.embed_dim)))
    
        #return (ug, vg), uh, u, v, x_recon
        return uh, u, v, x_recon
    def freeze_common(self, freeze):
        for name, p in self.encoder_c1.named_parameters():
            p.requires_grad = freeze
        for name, p in self.encoder_c2.named_parameters():
            p.requires_grad = freeze
    def see_freeze(self):
        for name, p in self.convEncoder.named_parameters():
            if p.requires_grad == True:
                return True
        for name, p in self.convDecoder.named_parameters():
            if p.requires_grad == True:
                return True
        return False
    def freeze_conv(self, freeze):
        for name, p in self.convEncoder.named_parameters():
            p.requires_grad = freeze
        for name, p in self.convDecoder.named_parameters():
            p.requires_grad = freeze
        #for name, p in self.convEncoder2.named_parameters():
            #p.requires_grad = freeze
class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 8, input_size = (256, 256)):
        super(ConvolutionalEncoder, self).__init__()
        #assuming iamge dim 256x256
        kernel_size = 3
        #right padding size to counteract kernel. 
        pad = math.floor((kernel_size - 1)/2)

        self.convEncoder1 = nn.Sequential(
            #need to choose stride and kernel size so that the dimensions decrease accordingly. 
            nn.Conv2d(in_channels, out_channels, kernel_size , stride = 1, padding = pad), nn.ReLU(), 
            nn.MaxPool2d(kernel_size, stride = 2, padding = pad), 
            nn.Conv2d(out_channels, 2*out_channels, kernel_size , stride = 1, padding = pad), nn.ReLU(), 
            nn.MaxPool2d(kernel_size, stride = 2, padding = pad),
            nn.Conv2d(2*out_channels, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.MaxPool2d(kernel_size, stride = 2, padding = pad),

            nn.Conv2d(4*out_channels, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.MaxPool2d(kernel_size , stride = 2, padding = pad),
            nn.Conv2d(4*out_channels, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.MaxPool2d(kernel_size, stride = 2, padding = pad),
            #one extra so 4x4
            nn.Conv2d(4*out_channels, 4*out_channels, kernel_size , stride = 1, padding = pad), nn.ReLU(), 
            nn.MaxPool2d(kernel_size, stride = 2, padding = pad), 
            nn.Conv2d(4*out_channels, 1, 1),
            nn.Flatten(start_dim = 1)
            #with new flatten change should be 8x8 = 64
        )

    def forward(self, x):
        return self.convEncoder1(x)
class ConvolutionalDecoder(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 8, input_size = (256, 256)):
        super(ConvolutionalDecoder, self).__init__()
        #assuming iamge dim 256x256
        kernel_size = 3
        #right padding size to counteract kernel. 
        pad = math.floor((kernel_size - 1)/2)

        self.convDecoder1 = nn.Sequential(
            #had just a conv2d here before, maybe was a problem. 4/2
            nn.ConvTranspose2d(1, 4*out_channels, 1), 
            nn.Upsample( scale_factor = 2), 
            nn.ConvTranspose2d(out_channels*4, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            #the above is the extra to make 4x4
            nn.Upsample( scale_factor = 2), 
            nn.ConvTranspose2d(out_channels*4, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.Upsample( scale_factor = 2), 
            nn.ConvTranspose2d(out_channels*4, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.Upsample( scale_factor = 2), 
            nn.ConvTranspose2d(out_channels*4, 2*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.Upsample( scale_factor = 2), 
            nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
            nn.Upsample(scale_factor =  2), 
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride = 1, padding = pad)
        )


    def forward(self, x):
        return self.convDecoder1(x)
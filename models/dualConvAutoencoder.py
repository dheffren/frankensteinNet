import torch
import torch.nn as nn
import math

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, model_cfg, loss_fn, hyp_sched,  metadata, device = "cpu"):
        #what? 
        super(ConvolutionalAutoencoder, self).__init__()
        #assuming iamge dim 256x256

        #get this from the dataset. 
        #TODO: Check if metadata is None what happens. 
        #if i call something.get what happens if object is none? 
        self.BW = metadata.get("BW", False) or False
        print(self.BW)
        #maybe these should be from the model config instead? 
        latent_dimU = metadata.get("latent_dimU", 2)
        latent_dimV = metadata.get("latent_dimV", 2)
        latent_dimC = metadata.get("latent_dimC", 2)
        in_channels = 3
        if self.BW:
            in_channels = 1
        #TODO: Make model construction depend on the image size. 
        self.data_size = metadata.get("input_shape", (in_channels, 256, 256))

        out_channels = model_cfg.get("out_channels", 8)
        kernel_size = model_cfg.get("kernel_size", 3)
        indim = model_cfg.get("indim", 40)
        dedim = model_cfg.get("dedim", 40)
        self.loss_fn = loss_fn
        self.hyp_sched = hyp_sched
        self.device = device

        self.convEncoder1 = ConvolutionalEncoder(in_channels, out_channels, kernel_size, self.data_size[1:])
        self.convEncoder2= ConvolutionalEncoder(in_channels, out_channels, kernel_size, self.data_size[1:])
        self.convDecoder1 = ConvolutionalDecoder(in_channels, out_channels, kernel_size, self.data_size[1:])
        self.convDecoder2 = ConvolutionalDecoder(in_channels, out_channels, kernel_size, self.data_size[1:])
        #need to relate this to the number of layers somehow. 
       
        #doing PCA with 16 features basically - perhaps too small? 
        #don't know how to deal with this with the current framework. 
        self.embed_dim = 8

        self.project_u1 = nn.Sequential(
            nn.Linear(self.embed_dim * self.embed_dim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim),
            nn.Linear(indim, latent_dimU)
        )
        self.project_u2 = nn.Sequential(
            nn.Linear(self.embed_dim*self.embed_dim,  indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim),
            nn.Linear(indim, latent_dimV)
        )
        self.project_c1 = nn.Sequential(
            nn.Linear(self.embed_dim*self.embed_dim,  indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim),
            nn.Linear(indim, latent_dimC)
        )
        self.project_c2 = nn.Sequential(
            nn.Linear(self.embed_dim*self.embed_dim,  indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim), nn.ReLU(), 
            nn.Linear(indim, indim),
            nn.Linear(indim, latent_dimC)
        )
   
        #this is RIDICULOUS
        self.deproject_1 = nn.Sequential(
            nn.Linear(latent_dimU + latent_dimC, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim),
            nn.Linear(dedim,self.embed_dim*self.embed_dim)
        )
        self.deproject_2 = nn.Sequential(
            nn.Linear(latent_dimC + latent_dimV, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim), nn.ReLU(), 
            nn.Linear(dedim, dedim),
            nn.Linear(dedim,self.embed_dim*self.embed_dim)
        )
    def loadEncDec(self):
        state_dictEnc = torch.load("trained_cae_encoder.pth")
        state_dictDec = torch.load("trained_cae_decoder.pth")
        self.convEncoder.load_state_dict(state_dictEnc)
        self.convDecoder.load_state_dict(state_dictDec)

    def outputImage1(self, u, c2): 
        uv = torch.cat([u,c2], dim=-1)
        uc = self.deproject_1(uv)
        x_recon = self.convDecoder1(uc.view(-1, 1, self.embed_dim, self.embed_dim))
        return x_recon
    def outputImage2(self, v, c1):
        uv = torch.cat([v, c1], dim=-1)
        uc = self.deproject_2(uv)
        x_recon = self.convDecoder2(uc.view(-1, 1, self.embed_dim, self.embed_dim))
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
    
    def forward(self, x1, x2):
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        #NOTE: When Have those extra layers but take a gradient skipping over those, weird things happen. 
        uh1 = self.convEncoder1(x1)  # Encode S1 (Rotations)
        print(f"uh1 shape: {uh1.shape}")
        uh2 = self.convEncoder2(x2)
        print(f"uh2 shape: {uh2.shape}")
        #print(uh1.shape)
        #print(uh2.shape)
 
       
        u = self.project_u1(uh1)


        #print("u: ", u.shape)
        c1 = self.project_c1(uh1)
        #print("C1: ", c1.shape)
        v = self.project_u2(uh2)
        #print("V: ", v.shape)
        c2 = self.project_c2(uh2)
        #print("C2: ", c2.shape)

        #Twisted architecture here. 
        uc1 = torch.cat([u,c2], dim=-1)
        uc2 = torch.cat([v,c1], dim =-1)
        #normal arch
        #uc1 = torch.cat([u,c1], dim = -1)
        #uc2 = torch.cat([v, c2], dim=-1)
        cu1 = self.deproject_1(uc1)
        cu2= self.deproject_2(uc2)
        #print("cu1:", cu1.shape)
        #print("cu2: ", cu2.shape)
        x_recon1 = self.convDecoder1(cu1.view((-1, 1, self.embed_dim, self.embed_dim)))
        x_recon2 = self.convDecoder2(cu2.view((-1, 1, self.embed_dim, self.embed_dim)))
        #print(x_recon1.shape)
        #print(x_recon2.shape)
        #return (ug, vg), uh, u, v, x_recon
        output_dict = {
            "recon": {"x1": x_recon1, "x2": x_recon2}, 
            "latent": {"latentUh1": uh1, 
        "latentUh2": uh2, 
        "latentU1": u, 
        "latentU2": v, 
        "latentC1": c1, 
        "latentC2": c2}, 
        }
        return output_dict 
    def compute_loss(self, batch, epoch):
        
        inputs, targets = self.prepare_input(batch)
        out = self(**inputs)
        #Update scheduled hyperparameters 
        lr1 = self.hyp_sched.get("lr1", epoch)
        lr2 = self.hyp_sched.get("lr2", epoch)
        lc = self.hyp_sched.get("lc", epoch)
        lo1 = self.hyp_sched.get("lo1", epoch)
        lo2 = self.hyp_sched.get("lo2", epoch)
        return self.loss_fn(out, targets, lr1, lr2, lc, lo1, lo2)
    def prepare_input(self, batch):
        #not sure how this generalizes at ALL to what I was doing before - with one input at a time. 
        x1, x2 = batch
        inputs = {"x1": x1.to(self.device).requires_grad_(), "x2": x2.to(self.device).requires_grad_()}
        targets = {"x1": x1, "x2": x2}
        return inputs, targets
    def freeze_common(self, freeze):
        for name, p in self.project_c1.named_parameters():
            p.requires_grad = freeze
        for name, p in self.project_c2.named_parameters():
            p.requires_grad = freeze
    def see_freeze(self):
        for name, p in self.convEncoder1.named_parameters():
            if p.requires_grad == True:
                return True
        for name, p in self.convEncoder1.named_parameters():
            if p.requires_grad == True:
                return True
        return False
    def freeze_conv(self, freeze):
        for name, p in self.convEncoder1.named_parameters():
            p.requires_grad = freeze
        for name, p in self.convEncoder2.named_parameters():
            p.requires_grad = freeze

        # don't freeze decoder. 
        #for name, p in self.convDecoder.named_parameters():
            #p.requires_grad = freeze
        #for name, p in self.convEncoder2.named_parameters():
            #p.requires_grad = freeze
class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 8, kernel_size = 3, input_size = (256, 256)):
        super(ConvolutionalEncoder, self).__init__()
        #assuming iamge dim 256x256
   
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
            #nn.Conv2d(4*out_channels, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
           # nn.MaxPool2d(kernel_size, stride = 2, padding = pad),
            #one extra so 4x4
            nn.Conv2d(4*out_channels, 1, 1),
            nn.Flatten(start_dim = 1)
            #with new flatten change should be 8x8 = 64
        )

    def forward(self, x):
        return self.convEncoder1(x)
class ConvolutionalDecoder(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 8, kernel_size = 3, input_size = (256, 256)):
        super(ConvolutionalDecoder, self).__init__()
        #assuming iamge dim 256x256

        #right padding size to counteract kernel. 
        pad = math.floor((kernel_size - 1)/2)

        self.convDecoder1 = nn.Sequential(
            #had just a conv2d here before, maybe was a problem. 4/2
            nn.ConvTranspose2d(1, 4*out_channels, 1), 
            #the above is the extra to make 4x4
            #nn.Upsample( scale_factor = 2), 
            #nn.ConvTranspose2d(out_channels*4, 4*out_channels, kernel_size, stride = 1, padding = pad), nn.ReLU(), 
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
    

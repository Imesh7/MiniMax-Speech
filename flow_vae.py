from torch import nn
import torch


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z)
        return recon_x

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


class Encoder(nn.Module):
    def __init__(self, in_features=512, out_features=128):
        super(Encoder, self).__init__()
        self.layer_1 = nn.Linear(in_features, 256)
        self.act1 = nn.ReLU()
        self.layer_2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()

        self.mean = nn.Linear(256, out_features)
        self.log_var = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act1(x)
        x = self.layer_2(x)
        x = self.act2(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, in_features=128, out_features=512):
        super(Decoder, self).__init__()
        self.layer_1 = nn.Linear(in_features, 256)
        self.act1 = nn.ReLU()
        self.layer_2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act1(x)
        x = self.layer_2(x)
        return x
    
    
def vae_loss(recon_x, x, mean, log_var):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_loss
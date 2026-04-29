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
    flow_loss = 0  # Placeholder for flow loss, if applicable
    return recon_loss + kl_loss

class VariationalDequantize(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = torch.distributions.normal.Normal(loc=0, scale=1.0)
        
    def encode(self, x):
        log_det_jacobian = 0
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_jacobian += log_det
        return x, log_det_jacobian
        
    def _get_likelihood(self, z):
        x, _ = self.encode(z)
        log_prob = self.prior.log_prob(x).sum(dim=-1)
        return log_prob
    
    def forward(self, x):
        log_likelihood = self._get_likelihood(x)
        return log_likelihood
    
    def sanmple(self, num_samples):
        z = self.prior.sample((num_samples,))
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z
    
class CouplingLayerFlow(nn.Module):
    def __init__(self,dim_in , hidden_dim):
        super().__init__()
        self.split_idx = dim_in // 2
        self.translation = nn.Sequential(
            nn.Linear(self.split_idx, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.split_idx, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x):
        x1 = x[:, :self.split_idx]
        x2 = x[:, self.split_idx:]
        
        t = self.translation(x1)
        s = self.scale(x2)
        
        z1 = x1
        z2 = x2 * torch.exp(s) + t
        
        return torch.cat([z1, z2], dim=1), s.sum(dim=1)  # Placeholder for log determinant of Jacobian

    
    def inverse(self, z):
        x1 = z[:, :self.split_idx]
        x2 = z[:, self.split_idx:]
        
        t = self.translation(x1)
        s = self.scale(x2)
        
        z1 = x1
        z2 = (x2 - t) * torch.exp(-s)
        
        return torch.cat([z1, z2], dim=1)  # Placeholder for inverse transformation
from torch import nn
import torch
import numpy as np


class FlowVAE(nn.Module):
    def __init__(self, dim_in=128, hidden_dim=256, coupling_layers=4):
        super(FlowVAE, self).__init__()
        self.encoder = Encoder()
        self.audio_flow = AudioFlow(
            flows=[
                CouplingLayerFlow(dim_in=dim_in, hidden_dim=hidden_dim)
                for _ in range(coupling_layers)
            ]
        )
        self.decoder = Decoder()

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        log_prob, log_det = self.audio_flow(z)  # Apply flow to the latent variable
        recon_x = self.decoder(z)
        return z, recon_x, mean, log_var, log_prob, log_det

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std


class Encoder(nn.Module):
    def __init__(self, in_features=512, out_features=128):
        super(Encoder, self).__init__()
        self.layer_1 = nn.Conv1d(in_features, out_features, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.layer_2 = nn.Conv1d(out_features, out_features, kernel_size=3, padding=1)
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


def vae_loss(z, recon_x, x, mean, log_var, log_det, epsilon):
    recon_loss = nn.MSELoss()(recon_x, x)
    log_q_z0 = -0.5 * torch.sum(
        log_var + (z - mean).pow(2) / log_var.exp() + np.log(2 * np.pi)
    )
    log_p_epsilon = -0.5 * torch.sum(epsilon.pow(2) + np.log(2 * np.pi), dim=-1)

    kl_loss = torch.sum(log_q_z0 - log_det - log_p_epsilon)

    return recon_loss + kl_loss


class AudioFlow(nn.Module):
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
        x, log_det = self.encode(z)
        log_prob = self.prior.log_prob(x).sum(dim=-1)
        return log_prob, log_det

    def forward(self, x):
        log_likelihood, log_det = self._get_likelihood(x)
        return log_likelihood, log_det

    @torch.no_grad()
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z


class CouplingLayerFlow(nn.Module):
    def __init__(self, dim_in, hidden_dim):
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
        x1 = x[:, : self.split_idx]
        x2 = x[:, self.split_idx :]

        t = self.translation(x1)
        s = self.scale(x1)

        z1 = x1
        z2 = x2 * torch.exp(s) + t

        return torch.cat([z1, z2], dim=1), s.sum(
            dim=1
        )  # Placeholder for log determinant of Jacobian

    def inverse(self, z):
        x1 = z[:, : self.split_idx]
        x2 = z[:, self.split_idx :]

        t = self.translation(x1)
        s = self.scale(x1)

        z1 = x1
        z2 = (x2 - t) * torch.exp(-s)

        return torch.cat([z1, z2], dim=1)

import math

from torch import nn
import torch

class SinoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        device = t.device
        half_dim = self.embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / half_dim))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class FlowMatchingModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.transformer_block = TransformerBlock(out_channels, nhead=4)
        self.time_embedding = nn.Sequential(
            SinoidalTimeEmbedding(embedding_dim=out_channels),
            nn.Linear(1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.velocity_proj = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        
        x_t = self.time_embedding(x)
        x = self.transformer_block(x, t=x_t)
        v = self.velocity_proj(v)
        return v


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Multi-head self-attention and feedforward layers
        self.multi_head_atten = nn.MultiheadAttention(d_model, nhead)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, t):
        attn_output, _ = self.multi_head_atten(x, x, x)
        
        x = self.norm1(attn_output)

        x = x + self.feedforward(x)

        x = self.norm2(x)

        return x

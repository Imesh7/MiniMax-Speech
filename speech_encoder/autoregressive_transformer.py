import torch
import torch.nn as nn


def time_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal time embeddings.

    Args:
        timesteps (torch.Tensor): A tensor of shape (batch_size,) containing the time steps.
        embedding_dim (int): The dimension of the time embeddings.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing the time embeddings.
    """
    half_dim = embedding_dim // 2
    emb = torch.exp(
        torch.arange(half_dim) * -(torch.log(torch.tensor(10000.0)) / half_dim)
    )
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
    return emb


class AutoregressiveTransformer:
    def __init__(self):
        self.text_embedding = nn.Embedding(num_embeddings=10000, embedding_dim=512)
        self.multi_head_self_attention = MultiHeadSelfAttention(d_in=512, d_out=512, num_heads=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )
        self.time_embedding_dim = 512

    def forward(self, features, tokens):
        text_emb = self.text_embedding(tokens)
        time_emb = time_embedding(features, self.time_embedding_dim)
        
        attn_out = x + self.multi_head_self_attention(text_emb + time_emb)
        x = x + self.feed_forward(attn_out)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x - is has the shape (batch_size, num_tokens, emb_dim)
        it combined of  'text_tokens' & 'audio_features'
        """
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)  # (batch_size, num_tokens, d_out * 3)

        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, num_heads, num_tokens, head_dim)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Each has shape (batch_size, num_heads, num_tokens, head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (batch_size, num_heads, num_tokens, num_tokens)
        attn_mask = torch.tril(torch.ones(num_tokens, num_tokens)).to(
            x.device
        )  # (num_tokens, num_tokens)
        attn_scores = attn_scores.masked_fill(
            attn_mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ v
        
        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec

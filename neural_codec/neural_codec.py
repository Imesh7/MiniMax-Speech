import torch
from torch.compiler import F
import torch.nn as nn


class NeuralCodec(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(
            input_dim, hidden_dim, num_embeddings=1000, embedding_dim=hidden_dim
        )
        self.vector_quantizer = VectorQuantizer(latent_dim=hidden_dim, k_size=1000)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, loss = self.vector_quantizer(z_e)
        output = self.decoder(z_q)
        return output, loss


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_1 = nn.Conv1d(embedding_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.embedding(x)
        return self.layer_1(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer_1 = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor):
        return self.layer_1(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, emb_dim, commitment_weight=1.0):
        super().__init__()
        self.embedding_dim = emb_dim
        self.codebook = nn.Embedding(num_embeddings, emb_dim)
        self.codebook.weight.data.uniform_(-1 / emb_dim, 1 / emb_dim)
        self.commitment_weight = commitment_weight

    def forward(self, z_e: torch.Tensor):
        (batch_size, time_seq, num_frames) = z_e.shape
        z_e = z_e.reshape(batch_size * time_seq, num_frames)

        # In here we needs to calculate the distance between z_e and
        # the codebook embeddings, and then find the closest embedding
        # for each vector in z_e.

        # This MSE equation is derived from the formula for the squared Euclidean distance:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        distance = (
            torch.sum(z_e**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight.t() ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(z_e, self.codebook.weight.t())
        )
        encoding_distances = torch.argmin(distance, dim=-1)
        z_q = self.codebook(encoding_distances)

        z_q = z_q.reshape(batch_size, time_seq, num_frames)

        # first mse loss is codebook loss
        # second part is commitment loss with a weight to balance the two losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment_weight * F.mse_loss(z_q.detach(), z_e)
        loss = codebook_loss + commitment_loss
        return z_q, loss

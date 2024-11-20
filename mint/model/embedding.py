import torch
from dataclasses import dataclass
import math


@dataclass
class EmbeddingConfig:
    vocab_size: int = "${..vocab_size}"
    d_model: int = "${glob.d_model}"
    max_seq_len: int = "${glob.max_seq_len}"
    learnable_positional_embeddings: bool = True


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, learnable_positional_embeddings):
        super(Embedding, self).__init__()
        self.learnable_positional_embeddings = learnable_positional_embeddings

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)

        if self.learnable_positional_embeddings:
            self.positional_embedding = torch.nn.Embedding(max_seq_len, d_model)
        else:
            self.register_buffer("positional_embedding", self.create_sinusoidal_embeddings(max_seq_len, d_model))
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.learnable_positional_embeddings:
            torch.nn.init.normal_(self.positional_embedding.weight, std=0.02)

    @staticmethod
    def create_sinusoidal_embeddings(max_length, embedding_dim):
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe

    def forward(self, x):
        """
        Args:
            x: torch.Tensor, shape (batch_size, seq_len)
        """
        B, S = x.size()
        t_emb = self.token_embedding(x)
        if self.learnable_positional_embeddings:
            p_emb = self.positional_embedding(torch.arange(S, device=x.device))
        else:
            p_emb = self.positional_embedding[:, :S, :]
        return t_emb + p_emb

import torch
import math
from dataclasses import dataclass


@dataclass
class MultiHeadAttentionConfig:
    n_heads: int = "${glob.n_heads}"
    d_model: int = "${glob.d_model}"
    max_seq_len: int = "${glob.max_seq_len}"
    context_window: int | None = "${glob.context_window}"


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, d_model, max_seq_len, context_window=None, causal=False):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.context_window = context_window
        self.causal = causal

        self.q_proj = torch.nn.Linear(self.d_model, self.d_model)
        self.k_proj = torch.nn.Linear(self.d_model, self.d_model)
        self.v_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.out_proj = torch.nn.Linear(self.d_model, self.d_model)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool), diagonal=1).unsqueeze(
                0).unsqueeze(0)  # upper triangular matrix filled with True
        )
        self.register_buffer(
            "context_mask",
            self.generate_context_mask(causal)
        )

        self.softmax = torch.nn.Softmax(dim=-1)

    def generate_context_mask(self, causal):
        if self.context_window:
            mask = torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
            for i in range(self.context_window):
                mask[i:].fill_diagonal_(0)

            if not causal:
                for i in range(self.context_window):
                    mask[:, i:].fill_diagonal_(0)
        else:
            mask = torch.zeros(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        return mask

    def forward(self, query, key_value=None):
        """
        Args:
            query: torch.Tensor, shape (batch_size, seq_len, d_model)
            key_value: torch.Tensor, shape (batch_size, seq_len, d_model)
            mask: torch.Tensor, shape (batch_size, seq_len)
        """
        if key_value is None:
            # self-attention
            key_value = query

        B, S_query, D = query.size()
        _, S_key_value, _ = key_value.size()
        D_head = D // self.n_heads  # dimension of single head

        # split into multiple heads, and exchange seq_len and n_heads dimensions
        q = self.q_proj(query).view(B, S_query, self.n_heads, D_head).transpose(1, 2)  # (B, n_heads, S, D // n_heads)
        k = self.k_proj(key_value).view(B, S_key_value, self.n_heads, D_head).transpose(1,
                                                                                        2)  # (B, n_heads, S_key_value, D // n_heads)
        v = self.v_proj(key_value).view(B, S_key_value, self.n_heads, D_head).transpose(1,
                                                                                        2)  # (B, n_heads, S_key_value, D // n_heads)

        # compute attention
        attention = q @ k.transpose(-2, -1)  # (B, n_heads, S_query, S_key_value)
        # scale the attention to stabilize training - taken from minigpt https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L62C43-L62C72
        attention = attention * (1.0 / math.sqrt(D_head))

        if self.causal:
            attention = attention.masked_fill(self.causal_mask[:, :, :S_query, :S_key_value], float('-inf'))

        # test: fill mask so that attention can look only at one previous token
        attention[:, :, self.context_mask[:S_query, :S_key_value]] = float('-inf')

        attention = self.softmax(attention)

        # aggregate the values based on the attention weights
        out = attention @ v

        # join the heads
        out = out.transpose(1, 2).contiguous().view(B, S_query, D)

        out = self.out_proj(out)

        return out

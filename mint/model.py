import torch
import math


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


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_feedforward, p_dropout, attention, causal=False):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.p_dropout = p_dropout

        self.multi_head_attention = MultiHeadAttention(causal=causal, **attention)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_feedforward),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_feedforward, self.d_model)
        )
        self.layer_norm_feedforward = torch.nn.LayerNorm(self.d_model)
        self.layer_norm_attention = torch.nn.LayerNorm(self.d_model)

        self.dropout = torch.nn.Dropout(self.p_dropout)

    def forward(self, x, y=None):
        """
        Args:
            x: torch.Tensor, shape (batch_size, seq_len, d_model)
        """

        x = x + self.dropout(self.multi_head_attention(self.layer_norm_attention(x), y))
        x = x + self.dropout(self.feed_forward(self.layer_norm_feedforward(x)))

        return x


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

    def create_sinusoidal_embeddings(self, max_length, embedding_dim):
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


class Encoder(torch.nn.Module):
    def __init__(self, n_blocks, vocab_size, transformer_block, embedding):
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks

        self.embedding = Embedding(**embedding)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(**transformer_block) for _ in range(self.n_blocks)]
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, n_blocks, vocab_size, d_model, d_feedforward, transformer_block, embedding):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size  # add one for the start token
        self.d_feedforward = d_feedforward

        embedding["vocab_size"] = self.vocab_size + 1
        self.embedding = Embedding(**embedding)
        self.decoder_start_token = self.vocab_size

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(causal=bool(i%2 == 0), **transformer_block)
            # twice as many because every decoder block consists of two sub-blocks (self-attention and encoder-decoder attention)
            for i in range(self.n_blocks * 2)
        ])

        self.vocabulary_projection = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_feedforward, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_feedforward, self.vocab_size, bias=False)
        )

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, decoder_tokens, encoder_output):
        # add decoder_start_token to the beginning of the input
        B = encoder_output.size(0)

        x = torch.full((B, 1), self.decoder_start_token, dtype=torch.long, device=encoder_output.device)
        if decoder_tokens is not None:
            x = torch.cat([x, decoder_tokens], dim=1)

        x = self.embedding(x)
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # self-attention
                x = block(x)
            else:
                # cross-attention
                x = block(x, encoder_output)

        logits = self.vocabulary_projection(x)
        return logits


class Transformer(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)

    def forward(self, encoder_tokens, decoder_tokens):
        encoder_output = self.encoder(encoder_tokens)
        decoder_output = self.decoder(decoder_tokens, encoder_output)

        return decoder_output

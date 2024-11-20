import torch
from dataclasses import dataclass,field

from mint.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
from mint.model.embedding import Embedding, EmbeddingConfig


@dataclass
class GlobalConfig:
    d_model: int = 0
    n_heads: int = 0
    max_seq_len: int = 0
    d_feedforward: int = 0
    p_dropout: float = 0.
    context_window: int | None = None


@dataclass
class TransformerBlockConfig:
    d_model: int = "${glob.d_model}"
    d_feedforward: int = "${glob.d_feedforward}"
    p_dropout: float = "${glob.p_dropout}"

    attention_config: MultiHeadAttentionConfig = field(default_factory=MultiHeadAttentionConfig)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, d_feedforward, p_dropout, attention_config, causal=False):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.p_dropout = p_dropout

        self.multi_head_attention = MultiHeadAttention(causal=causal, **attention_config)
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



@dataclass
class EncoderConfig:
    n_blocks: int = 0
    vocab_size: int = 0
    transformer_block_config: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)


class Encoder(torch.nn.Module):
    def __init__(self, n_blocks, vocab_size, transformer_block_config, embedding_config):
        super(Encoder, self).__init__()
        self.n_blocks = n_blocks

        self.embedding = Embedding(**embedding_config)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(**transformer_block_config) for _ in range(self.n_blocks)]
        )

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class DecoderConfig:
    n_blocks: int = 0
    vocab_size: int = 0
    d_model: int = "${glob.d_model}"
    d_feedforward: int = "${glob.d_feedforward}"
    transformer_block_config: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)


class Decoder(torch.nn.Module):
    def __init__(self, n_blocks, vocab_size, d_model, d_feedforward, transformer_block_config, embedding_config):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.vocab_size = vocab_size  # add one for the start token
        self.d_feedforward = d_feedforward

        embedding_config["vocab_size"] = self.vocab_size + 1
        self.embedding = Embedding(**embedding_config)
        self.decoder_start_token = self.vocab_size

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(causal=bool(i%2 == 0), **transformer_block_config)
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


@dataclass
class TransformerConfig:
    glob: GlobalConfig = field(default_factory=GlobalConfig)

    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)


class Transformer(torch.nn.Module):
    def __init__(self, encoder_config, decoder_config, glob=None):
        super(Transformer, self).__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

    def forward(self, encoder_tokens, decoder_tokens):
        encoder_output = self.encoder(encoder_tokens)
        decoder_output = self.decoder(decoder_tokens, encoder_output)

        return decoder_output

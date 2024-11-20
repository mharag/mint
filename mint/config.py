from omegaconf import OmegaConf
from dataclasses import dataclass, field
from functools import partial


@dataclass
class ModelConfig:
    d_model: int = 0
    n_heads: int = 0
    max_seq_len: int = 0
    d_feedforward: int = 0
    p_dropout: float = 0.
    context_window: int | None = None


@dataclass
class MultiHeadAttentionConfig:
    n_heads: int = "${model.n_heads}"
    d_model: int = "${model.d_model}"
    max_seq_len: int = "${model.max_seq_len}"
    context_window: int | None = "${model.context_window}"


@dataclass
class TransformerBlockConfig:
    d_model: int = "${model.d_model}"
    d_feedforward: int = "${model.d_feedforward}"
    p_dropout: float = "${model.p_dropout}"

    attention: MultiHeadAttentionConfig = field(default_factory=MultiHeadAttentionConfig)


@dataclass
class EmbeddingConfig:
    vocab_size: int = "${..vocab_size}"
    d_model: int = "${model.d_model}"
    max_seq_len: int = "${model.max_seq_len}"
    learnable_positional_embeddings: bool = True


@dataclass
class EncoderConfig:
    n_blocks: int = 0
    vocab_size: int = 0
    transformer_block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class DecoderConfig:
    n_blocks: int = 0
    vocab_size: int = 0
    d_model: int = "${model.d_model}"
    d_feedforward: int = "${model.d_feedforward}"
    transformer_block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class Transformer:
    model: ModelConfig = field(default_factory=ModelConfig)

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


config = OmegaConf.structured(Transformer())


def to_dict(config):
    config_dict = OmegaConf.to_container(config, resolve=True)
    # filter model out
    return {k: v for k, v in config_dict.items() if k != "model"}


def get_config(config):
    return OmegaConf.structured(config)


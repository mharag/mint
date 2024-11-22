from dataclasses import dataclass
from transformers import AutoTokenizer
import os
from mint.common import create_config, load_yaml_config, save_yaml_config


@dataclass
class TokenizerConfig:
    vocab_size: int = 0
    architecture: str = "gpt2"
    pad_token: str = "<pad>"


class Tokenizer:
    def __init__(self, vocab_size, architecture, pad_token):
        self.vocab_size = vocab_size
        self.architecture = architecture
        self.pad_token = pad_token

        self.tokenizer = None

    def train(self, dataset_path):
        pretrained = AutoTokenizer.from_pretrained("gpt2")
        dataset = (line for line in open(dataset_path))
        self.tokenizer = pretrained.train_new_from_iterator(dataset, show_progress=True, vocab_size=self.vocab_size)
        self.tokenizer.pad_token = self.pad_token

    def save(self, path):
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        config = create_config(TokenizerConfig(
            vocab_size=self.vocab_size,
            architecture=self.architecture,
            pad_token=self.pad_token
        ))
        save_yaml_config(config, os.path.join(path, "tokenizer_config"))

    @classmethod
    def load(cls, path):
        config = load_yaml_config(os.path.join(path, "tokenizer_config"))
        tokenizer = cls(config.vocab_size, config.architecture, config.pad_token)
        tokenizer.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        return tokenizer


    def tokenize(self, text, max_length=128):
        return self.tokenizer(text, truncation=True, padding="max_length", max_length=max_length).input_ids


    def detokenize(self, tokens):
        return self.tokenizer.batch_decode(tokens)
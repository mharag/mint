from datasets import load_dataset, Dataset as HuggingFaceDataset, DatasetDict
import os
from transformers import AutoTokenizer
import random


class Dataset(DatasetDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_tokenizer = None
        self.target_tokenizer = None

    @classmethod
    def load_preprocessed(cls, path: str):
        dataset = cls.load_from_disk(os.path.join(path, "dataset"))
        dataset.source_tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "source_tokenizer"))
        dataset.target_tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "target_tokenizer"))
        return dataset

    @classmethod
    def preprocess(cls, source_input_file, target_input_file, output_dir, vocab_size, test_split, max_seq_len):
        source_dataset = load_dataset("text", data_files=source_input_file, split="train", streaming=True)
        target_dataset = load_dataset("text", data_files=target_input_file, split="train", streaming=True)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        source_tokenizer = tokenizer.train_new_from_iterator(source_dataset, show_progress=True, vocab_size=10000)
        target_tokenizer = tokenizer.train_new_from_iterator(target_dataset, show_progress=True, vocab_size=10000)
        source_tokenizer.save_pretrained(os.path.join(output_dir, "source_tokenizer"))
        target_tokenizer.save_pretrained(os.path.join(output_dir, "target_tokenizer"))
        source_tokenizer.pad_token = "<pad>"
        target_tokenizer.pad_token = "<pad>"

        def tokenize():
            for source, target in zip(source_dataset, target_dataset):
                source_tokens = source_tokenizer(source["text"], truncation=True, padding="max_length", max_length=max_seq_len)
                target_tokens = target_tokenizer(target["text"], truncation=True, padding="max_length", max_length=max_seq_len)
                yield {
                    "source_tokens": source_tokens.input_ids,
                    "target_tokens": target_tokens.input_ids,
                }

        tokenized_dataset = tokenize()

        # Split into train and test datasets as lists
        train_data = []
        test_data = []
        for i, example in enumerate(tokenized_dataset):
            if random.random() < test_split:
                test_data.append(example)
            else:
                train_data.append(example)

        # Convert the lists to Hugging Face Datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)

        # Combine into a DatasetDict
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        # Save the datasets to disk
        dataset_dict.save_to_disk(os.path.join(output_dir, "dataset"))

        print("Tokenized datasets saved successfully!")

    def tokenize(self, text, target=False, max_seq_len=None):
        if not target:
            tokenizer = self.source_tokenizer
        else:
            tokenizer = self.target_tokenizer
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_seq_len)




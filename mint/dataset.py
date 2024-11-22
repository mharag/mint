from datasets import load_dataset, Dataset as HuggingFaceDataset, DatasetDict
import os
from transformers import AutoTokenizer
import random
from tqdm import tqdm
from subprocess import check_output


class Dataset(DatasetDict):
    @classmethod
    def load_preprocessed(cls, path: str):
        dataset = cls.load_from_disk(os.path.join(path, "preprocessed_dataset"))
        return dataset

    @classmethod
    def preprocess(
        cls,
        source_input_file,
        target_input_file,
        source_tokenizer,
        target_tokenizer,
        output_dir,
        max_seq_len,
        test_split=0.1,
    ):
        source_dataset = load_dataset("text", data_files=source_input_file, split="train", streaming=True)
        target_dataset = load_dataset("text", data_files=target_input_file, split="train", streaming=True)

        def tokenize():
            for source, target in zip(source_dataset, target_dataset):
                source_tokens = source_tokenizer.tokenize(source["text"], max_length=max_seq_len)
                target_tokens = target_tokenizer.tokenize(target["text"], max_length=max_seq_len)
                yield {
                    "source_tokens": source_tokens.input_ids,
                    "target_tokens": target_tokens.input_ids,
                }

        tokenized_dataset = tokenize()

        # Split into train and test datasets as lists
        train_data = []
        test_data = []
        for item in tqdm(tokenized_dataset):
            if random.random() < test_split:
                test_data.append(item)
            else:
                train_data.append(item)

        # Convert the lists to Hugging Face Datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)

        # Combine into a DatasetDict
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

        # Save the datasets to disk
        dataset_dict.save_to_disk(os.path.join(output_dir, "dataset"))

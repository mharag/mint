from datasets import load_dataset, DatasetDict, Dataset as HuggingFaceDataset
import os
import random
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter


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
        buffer_size=1000,  # Number of examples to process in a chunk
    ):
        source_dataset = load_dataset("text", data_files=source_input_file, split="train", streaming=True)
        target_dataset = load_dataset("text", data_files=target_input_file, split="train", streaming=True)

        os.makedirs(output_dir, exist_ok=True)

        train_output_path = os.path.join(output_dir, "train_temp.arrow")
        test_output_path = os.path.join(output_dir, "test_temp.arrow")

        train_writer = ArrowWriter(path=train_output_path)
        test_writer = ArrowWriter(path=test_output_path)

        def tokenize_and_write():
            train_count, test_count = 0, 0  # Track written examples
            buffer = []

            for source, target in tqdm(zip(source_dataset, target_dataset), desc="Tokenizing and writing data"):
                source_tokens = source_tokenizer.tokenize(
                    source["text"], max_length=max_seq_len
                )
                target_tokens = target_tokenizer.tokenize(
                    target["text"], max_length=max_seq_len
                )

                item = {
                    "source_tokens": source_tokens,
                    "target_tokens": target_tokens,
                }
                buffer.append(item)

                if len(buffer) >= buffer_size:
                    for record in buffer:
                        if random.random() < test_split:
                            test_writer.write(record)
                            test_count += 1
                        else:
                            train_writer.write(record)
                            train_count += 1
                    buffer.clear()

            for record in buffer:
                if random.random() < test_split:
                    test_writer.write(record)
                    test_count += 1
                else:
                    train_writer.write(record)
                    train_count += 1

            return train_count, test_count

        train_count, test_count = tokenize_and_write()

        train_writer.finalize()
        test_writer.finalize()

        train_dataset = HuggingFaceDataset.from_file(train_output_path)
        test_dataset = HuggingFaceDataset.from_file(test_output_path)

        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
        dataset_dict.save_to_disk(os.path.join(output_dir, "dataset"))

        # Cleanup temporary files
        os.remove(train_output_path)
        os.remove(test_output_path)

        print(f"Data preprocessing completed: {train_count} train examples, {test_count} test examples.")

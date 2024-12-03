from datasets import DatasetDict, Dataset as HuggingFaceDataset, load_dataset
import os


class Dataset(DatasetDict):
    @classmethod
    def load(cls, dataset_dir: str):
        train_path = os.path.join(dataset_dir, "train.jsonl")
        test_path = os.path.join(dataset_dir, "test.jsonl")

        dataset = load_dataset(
            "json",
            data_files={"train": train_path, "test": test_path},
            streaming=True
        )
        return dataset
from zenml import step, pipeline
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer


class StarWarsDataset(Dataset):
    """Custom Dataset for Star Wars dialogue data."""

    def __init__(self, dataset_split=None, max_length=128):
        """Initialize the dataset."""
        full_dataset = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
        self.dataset = full_dataset["train"] if dataset_split is None else dataset_split
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx]["Line"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "character": self.dataset[idx]["Character"],
        }


def create_dataloaders(
    batch_size: int = 32, max_length: int = 128, train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    full_dataset = StarWarsDataset(max_length=max_length)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


@step(enable_cache=False)
def load_data(
    batch_size: int = 32, max_length: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """ZenML step to create and return DataLoaders."""
    return create_dataloaders(batch_size=batch_size, max_length=max_length)


@pipeline
def simple_ml_pipeline():
    """Define a pipeline that connects the steps."""

    load_data()


if __name__ == "__main__":
    run = simple_ml_pipeline()

    # You can now use the `run` object to see steps, outputs, etc.

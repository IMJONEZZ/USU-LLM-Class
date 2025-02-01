import pytest
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline import StarWarsDataset, create_dataloaders


def test_dataset_structure():
    """Test that we can access the dataset and its expected structure."""
    dataset = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    assert "train" in dataset
    sample = dataset["train"][0]
    assert "Line" in sample
    assert "Character" in sample


def test_dataset_initialization():
    """Test that the dataset can be initialized properly."""
    dataset = StarWarsDataset()
    assert dataset is not None
    assert len(dataset) > 0


def test_dataset_item_format():
    """Test that items returned by the dataset have the correct format."""
    dataset = StarWarsDataset()
    item = dataset[0]

    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "character" in item

    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["character"], str)
    assert item["input_ids"].dim() == 1
    assert item["attention_mask"].dim() == 1
    assert item["input_ids"].shape == item["attention_mask"].shape


def test_max_length_constraint():
    """Test that the max_length parameter is respected."""
    max_length = 50
    dataset = StarWarsDataset(max_length=max_length)
    item = dataset[0]

    assert item["input_ids"].shape[0] == max_length
    assert item["attention_mask"].shape[0] == max_length


def test_dataloader_creation():
    """Test that DataLoaders are created with the correct configuration."""
    batch_size = 16
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert train_loader.batch_size == batch_size
    assert val_loader.batch_size == batch_size


def test_batch_format():
    """Test that batches from the DataLoader have the correct format."""
    batch_size = 4
    train_loader, _ = create_dataloaders(batch_size=batch_size)

    batch = next(iter(train_loader))

    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "character" in batch
    assert batch["input_ids"].shape[0] == batch_size
    assert batch["attention_mask"].shape[0] == batch_size
    assert len(batch["character"]) == batch_size


def test_train_val_split():
    """Test that train and validation splits have correct proportions."""
    train_split = 0.8
    batch_size = 32
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size, train_split=train_split
    )

    total_train_samples = len(train_loader.dataset)
    total_val_samples = len(val_loader.dataset)
    total_samples = total_train_samples + total_val_samples

    assert abs(total_train_samples / total_samples - train_split) < 0.01


def test_attention_mask_validity():
    """Test that attention masks contain only 0s and 1s."""
    dataset = StarWarsDataset()
    item = dataset[0]

    mask_values = item["attention_mask"].unique()
    assert all(value in [0, 1] for value in mask_values.tolist())


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_various_batch_sizes(batch_size):
    """Test that different batch sizes work correctly."""
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)
    batch = next(iter(train_loader))

    # Test batch dimensions
    assert batch["input_ids"].shape[0] == batch_size
    assert batch["attention_mask"].shape[0] == batch_size
    assert len(batch["character"]) == batch_size

    # Check types for each batch element
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert isinstance(batch["character"], list)
    assert all(isinstance(char, str) for char in batch["character"])

    # Test shape consistency across batch
    first_seq_len = batch["input_ids"].shape[1]
    assert batch["attention_mask"].shape[1] == first_seq_len

    # Test for any invalid values in attention mask
    assert torch.all((batch["attention_mask"] == 0) | (batch["attention_mask"] == 1))

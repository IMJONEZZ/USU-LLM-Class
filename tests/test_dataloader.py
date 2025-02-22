import pytest
from torch.utils.data import DataLoader
from dataloader import TextDataset, create_dataloader

# Sample encoding data for testing
sample_encoding = {
    "input_ids": [
        [101, 2023, 2003, 1037, 7072, 102],
        [101, 2023, 2003, 1037, 7072, 102],
    ],
    "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
}


def test_text_dataset_length():
    """Test that the length of the dataset is correct."""
    dataset = TextDataset(sample_encoding)
    assert len(dataset) == 2


def test_text_dataset_get_item():
    """Test that the dataset returns the correct item."""
    dataset = TextDataset(sample_encoding)
    item = dataset[0]
    assert item["input_ids"] == [101, 2023, 2003, 1037, 7072, 102]
    assert item["attention_mask"] == [1, 1, 1, 1, 1, 1]


def test_create_dataloader():
    """Test that the DataLoader is created correctly."""
    dataloader = create_dataloader(sample_encoding, batch_size=2)
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == 2
    assert dataloader.shuffle is True

    # Check that the DataLoader returns the correct number of batches
    batches = list(dataloader)
    assert len(batches) == 1  # Since we have 2 samples and batch_size is 2

    # Check the content of the first batch
    batch = batches[0]
    assert batch["input_ids"].shape == (2, 6)  # 2 samples, each with 6 tokens
    assert batch["attention_mask"].shape == (2, 6)


def test_create_dataloader_with_different_batch_size():
    """Test that the DataLoader works with a different batch size."""
    dataloader = create_dataloader(sample_encoding, batch_size=1)
    assert dataloader.batch_size == 1

    # Check that the DataLoader returns the correct number of batches
    batches = list(dataloader)
    assert len(batches) == 2  # Since we have 2 samples and batch_size is 1

    # Check the content of the first batch
    batch = batches[0]
    assert batch["input_ids"].shape == (1, 6)  # 1 sample with 6 tokens
    assert batch["attention_mask"].shape == (1, 6)


def test_create_dataloader_with_empty_encoding():
    """Test that the DataLoader handles empty encoding correctly."""
    empty_encoding = {"input_ids": [], "attention_mask": []}
    dataset = TextDataset(empty_encoding)
    assert len(dataset) == 0

    dataloader = create_dataloader(empty_encoding, batch_size=2)
    assert len(list(dataloader)) == 0


def test_create_dataloader_with_invalid_encoding():
    """Test that the DataLoader raises an error with invalid encoding."""
    invalid_encoding = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1]],
    }  # Mismatched lengths
    with pytest.raises(ValueError):
        TextDataset(invalid_encoding)

import pytest
import torch
from torch.utils.data import DataLoader
from run import TokenDataset, collate_fn  # Import the dataset and collate_fn


@pytest.fixture
def sample_data():
    """Provides a sample tokenized dataset."""
    return [
        [1, 2, 3, 4, 5],  # Short sequence
        [6, 7, 8, 9],  # Another short sequence
        [10] * 50,  # Long sequence (tests max padding)
        [11, 12, 13],  # Another short sequence
    ]


@pytest.fixture
def dataloader(sample_data):
    """Creates a DataLoader instance using the sample data."""
    dataset = TokenDataset(sample_data)
    return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


def test_dataset_length(sample_data):
    """Test that dataset length matches number of sequences."""
    dataset = TokenDataset(sample_data)
    assert len(dataset) == len(sample_data), (
        f"Expected {len(sample_data)}, got {len(dataset)}"
    )


def test_collate_fn_padding(sample_data):
    """Ensure that all sequences in a batch are padded to exactly 50."""
    dataset = TokenDataset(sample_data)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, max_len=50),
    )

    batch = next(iter(dataloader))  # Get first batch
    expected_length = 50

    assert batch.shape[1] == expected_length, (
        f"Expected padded sequence length to be {expected_length}, but got {batch.shape[1]}"
    )


def test_dataloader_batching(sample_data):
    """Check if DataLoader correctly batches data to fixed length."""
    dataset = TokenDataset(sample_data)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, max_len=50),
    )

    batch = next(iter(dataloader))  # Get first batch
    expected_shape = (2, 50)  # (batch_size, max_len)

    assert batch.shape == expected_shape, (
        f"Expected batch shape {expected_shape}, got {batch.shape}"
    )


def test_dataloader_shuffle(sample_data):
    """Ensure that shuffling changes the order of sequences."""
    dataset = TokenDataset(sample_data)

    dataloader_shuffled = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    dataloader_unshuffled = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    shuffled_batches = list(iter(dataloader_shuffled))
    unshuffled_batches = list(iter(dataloader_unshuffled))

    assert any(
        not torch.equal(shuffled_batches[i], unshuffled_batches[i])
        for i in range(len(shuffled_batches))
    ), "Shuffling is not working as expected"

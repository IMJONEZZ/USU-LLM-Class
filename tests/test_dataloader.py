import pytest
import torch
from torch.utils.data import DataLoader
from your_module import TextDataset  # Import your dataset class

@pytest.fixture
def sample_dataset():
    txt = "0 1 2 3 4 5 6 7 8 9 10"
    tokenizer = None  # Assuming your dataset handles tokenization
    return TextDataset(txt, tokenizer, max_length=5, stride=2)

def test_dataloader_loads_data(sample_dataset):
    dataloader = DataLoader(sample_dataset, batch_size=2, shuffle=False)

    # Fetch the first batch
    batch = next(iter(dataloader))
    assert len(batch) == 2  # Should return (input_ids, target_ids)
    assert isinstance(batch[0], torch.Tensor)  # Input should be a tensor
    assert isinstance(batch[1], torch.Tensor)  # Target should be a tensor

def test_dataloader_batch_size(sample_dataset):
    dataloader = DataLoader(sample_dataset, batch_size=3, shuffle=False)
    batch = next(iter(dataloader))
    
    assert batch[0].shape[0] == 3  # Check if batch size is 3

def test_dataloader_shuffling(sample_dataset):
    dataloader1 = DataLoader(sample_dataset, batch_size=2, shuffle=True)
    dataloader2 = DataLoader(sample_dataset, batch_size=2, shuffle=True)

    # Extract batches
    batch1 = list(iter(dataloader1))
    batch2 = list(iter(dataloader2))

    # Check if order is different in at least one case
    assert batch1 != batch2  # Shuffled dataloaders should differ sometimes

@pytest.mark.parametrize("drop_last", [True, False])
def test_dataloader_drop_last(sample_dataset, drop_last):
    dataloader = DataLoader(sample_dataset, batch_size=4, drop_last=drop_last)
    
    num_batches = len(list(dataloader))
    expected_batches = len(sample_dataset) // 4 if drop_last else -(-len(sample_dataset) // 4)  # Ceiling division

    assert num_batches == expected_batches  # Ensure correct batch count

def test_empty_dataset():
    dataset = TextDataset("", None, max_length=5, stride=2)
    dataloader = DataLoader(dataset, batch_size=2)

    with pytest.raises(StopIteration):  # Should raise error when iterating
        next(iter(dataloader))
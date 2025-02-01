import pytest
import torch
from torch.utils.data import DataLoader
from main import TextDataset, create_dataloader_v1, SimpleTokenizer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Sample vocab
@pytest.fixture
def sample_vocab():
    return {"th": 0, "he": 1, "er": 2, "<|unk|>": 3}  # Small test vocab


# Sample dataset for testing
@pytest.fixture
def sample_text():
    return {"sentence": [0, 1, 2, 0, 1, 2, 0, 1, 2]}  # Tokenized indices


# Sample tokenizer
@pytest.fixture
def tokenizer(sample_vocab):
    return SimpleTokenizer(sample_vocab)


# Test Dataset Initialization
def test_textdataset_init(sample_text):
    dataset = TextDataset(sample_text["sentence"], max_length=4, stride=2)

    assert len(dataset) > 0, "Dataset should not be empty"
    assert isinstance(dataset[0], tuple), "Dataset items should be tuples"
    assert len(dataset[0]) == 2, "Each sample should contain input and target tensors"
    assert dataset[0][0].shape == dataset[0][1].shape, (
        "Input and target tensors should have same shape"
    )


# Test DataLoader Batch Shape
def test_dataloader_batch_shape(sample_text):
    dataloader = create_dataloader_v1(
        sample_text["sentence"], batch_size=2, max_length=4, stride=2
    )

    batch = next(iter(dataloader))  # Get first batch
    inputs, targets = batch

    assert isinstance(inputs, torch.Tensor), "Inputs should be a tensor"
    assert isinstance(targets, torch.Tensor), "Targets should be a tensor"
    assert inputs.shape[0] == 2, "Batch size should be 2"
    assert inputs.shape == targets.shape, "Input and target shapes should match"


# Test DataLoader Iteration
def test_dataloader_iteration(sample_text):
    dataloader = create_dataloader_v1(
        sample_text["sentence"], batch_size=2, max_length=4, stride=2
    )

    for batch in dataloader:
        inputs, targets = batch
        assert inputs.shape == targets.shape, (
            "Each batch's inputs and targets should match in shape"
        )


# Test Edge Case: Empty Data
def test_empty_dataset():
    dataset = TextDataset([], max_length=4, stride=2)
    assert len(dataset) == 0, "Dataset should be empty if no tokens are given"

    dataloader = create_dataloader_v1([], batch_size=2, max_length=4, stride=2)
    with pytest.raises(StopIteration):
        next(iter(dataloader))  # Should raise StopIteration since it's empty

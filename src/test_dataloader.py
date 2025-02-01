import torch
import pytest
from torch.utils.data import TensorDataset
from dataloader import CustomDataLoader

@pytest.fixture
def sample_dataset():
    """Creates a simple dataset for testing."""
    inputs = torch.arange(20).view(10, 2)  # 10 samples, each with 2 features
    labels = torch.arange(10)  # 10 labels
    return TensorDataset(inputs, labels)

def test_dataloader_initialization(sample_dataset):
    #this tests if the dataloader initializes properly -> PASSED
    dataloader = CustomDataLoader(sample_dataset, batch_size=2, shuffle=False)
    assert len(dataloader) == 5 #10 samples with batch_size 2

def test_dataloader_batch_size(sample_dataset):
    #tests if dataloader makes correct batch size
    batch_size = 3
    dataloader = CustomDataLoader(sample_dataset, batch_size=batch_size, shuffle=False)
    batches = list(dataloader)

    assert len(batches) == (len(sample_dataset) + batch_size - 1) // batch_size
    for batch in batches[:-1]:
        assert batch[0].shape[0] == batch_size
        assert batch[1].shape[0] == batch_size
if __name__ == "__main__":
    pytest.main()
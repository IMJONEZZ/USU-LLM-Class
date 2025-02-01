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
    #this tests if the dataloader initializes properly
    dataloader = CustomDataLoader(sample_dataset, batch_size=2, shuffle=False)
    assert len(dataloader) == 5 #10 samples with batch_size 2

if __name__ == "__main__":
    pytest.main()
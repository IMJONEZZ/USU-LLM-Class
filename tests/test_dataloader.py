import json
import pytest
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append("/Users/chenhaozhe/Desktop/LLM/USU-LLM-Class/HW3")
from HW3.run import SWDataset, load_data, data_process_pipeline


@pytest.fixture
def sample_data():
    """Fixture providing sample JSON-like data."""
    return [
        {"Character": "THREEPIO", "Line": "Did you hear that?"},
        {"Character": "VADER", "Line": "The Force is strong with this one."},
        {"Character": "LUKE", "Line": "I am a Jedi, like my father before me."},
    ]


### 1. Test SWDataset Initialization and Functionality
def test_swdataset_length(sample_data):
    """Ensure SWDataset correctly reports its length."""
    dataset = SWDataset(sample_data)
    assert len(dataset) == len(sample_data)


### 2.
def test_swdataset_getitem(sample_data):
    """Ensure SWDataset correctly retrieves items."""
    dataset = SWDataset(sample_data)
    assert dataset[0]["Character"] == "THREEPIO"
    assert dataset[1]["Line"] == "The Force is strong with this one."


### 3. Test the load_data Step


def test_load_data(tmp_path):
    """Test load_data with a real JSON file."""

    # Create a temporary JSON file
    json_file = tmp_path / "SW_EpisodeIV_VI.json"

    # Sample data
    sample_data = [
        {"Character": "THREEPIO", "Line": "Did you hear that?"},
        {"Character": "VADER", "Line": "The Force is strong with this one."},
    ]

    # Write sample data to the file
    json_file.write_text(json.dumps(sample_data))

    # Run load_data (it will read from the real file)
    data = load_data()

    # Validate DataLoader presence
    assert "dataloader" in data
    assert isinstance(data["dataloader"], DataLoader)

    # Validate DataLoader output
    dataloader = data["dataloader"]
    batches = list(dataloader)

    assert len(batches) > 0
    assert len(batches[0]) == 2  # Batch size is 2


### 4. Test the Full Pipeline Execution
def test_pipeline_execution():
    """Ensure the full pipeline runs without errors."""
    try:
        data_process_pipeline()
    except Exception as e:
        pytest.fail(f"Pipeline execution failed with error: {e}")


### 5. Test DataLoader Batch Size
def test_dataloader_batch_size(sample_data):
    """Ensure DataLoader returns batches of the correct size."""
    dataset = SWDataset(sample_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batches = list(dataloader)
    assert all(len(batch) == 2 for batch in batches[:-1])  # Last batch may be smaller


### 6. Test DataLoader Shuffling
def test_dataloader_shuffling(sample_data):
    """Ensure DataLoader correctly shuffles data."""
    dataset = SWDataset(sample_data)
    dataloader_1 = DataLoader(dataset, batch_size=2, shuffle=True)
    dataloader_2 = DataLoader(dataset, batch_size=2, shuffle=True)

    # Extract first batch from two different runs
    first_batch_1 = next(iter(dataloader_1))
    first_batch_2 = next(iter(dataloader_2))

    assert first_batch_1 != first_batch_2, "Shuffling should result in different orders"

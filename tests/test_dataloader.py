# test_dataloader.py
import json
import pytest
from torch.utils.data import DataLoader
import sys

sys.path.append("C:/Users/tjker/Desktop/School/Spring_2025/USU-LLM-Class/usu_llm_class")
from usu_llm_class.data import StarWarsDataset


# Use pytest's tmp_path fixture to create a temporary JSON file with dummy data.
@pytest.fixture
def sample_dataset(tmp_path):
    # Create dummy data: a list of dialogue dictionaries.
    data = [
        {"Character": "LUKE", "Line": "I am your father?"},
        {"Character": "VADER", "Line": "No, I am your father."},
        {"Character": "HAN", "Line": "Never tell me the odds!"},
    ]
    json_file = tmp_path / "SW_EpisodeIV_VI.json"
    with open(json_file, "w") as f:
        json.dump(data, f)
    # Instantiate dataset using the temporary JSON file.
    return StarWarsDataset(str(json_file))


@pytest.fixture
def sample_dataloader(sample_dataset):
    return DataLoader(sample_dataset, batch_size=2, shuffle=False)


def test_dataset_length(sample_dataset):
    # Verify that the dataset length matches the number of dummy entries.
    assert len(sample_dataset) == 3


def test_dataloader_batch(sample_dataloader):
    # Get the first batch from the DataLoader.
    batch = next(iter(sample_dataloader))

    assert isinstance(batch, dict)
    assert "Character" in batch
    assert "Line" in batch

    characters = batch["Character"]
    lines = batch["Line"]
    assert len(characters) == 2  # because batch_size is 2
    assert len(lines) == 2


def test_full_iteration(sample_dataloader):
    all_batches = list(sample_dataloader)
    assert len(all_batches) == 2
    total_samples = sum(len(batch["Character"]) for batch in all_batches)
    assert total_samples == 3

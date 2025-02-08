import pytest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from evaluator import evaluate_model
from main import assignment_4_pipeline, TextDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from zenml.client import Client


@pytest.fixture
def mock_dataloader():
    """Fixture to create a mock DataLoader."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.eval()

    # Create mock data for the DataLoader
    mock_data = [
        {"Line": "This is a test sentence."},
        {"Line": "Another test sentence."},
    ]
    encoding = tokenizer(
        [entry["Line"] for entry in mock_data],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    dataset = TextDataset(encoding)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    return dataloader


def test_evaluate_model_success(mock_dataloader):
    """Test for successful model evaluation."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.eval()

    loss = evaluate_model(mock_dataloader, model)

    assert isinstance(loss, float), "Loss should be a float."
    assert loss >= 0, "Loss should be non-negative."


def test_evaluate_model_invalid_input():
    """Test for invalid input types."""
    with pytest.raises(TypeError):
        evaluate_model(None, None)  # Invalid DataLoader and model


def test_evaluate_model_loss(mock_dataloader):
    """Test to ensure that loss is returned and is a numeric value."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.eval()

    loss = evaluate_model(mock_dataloader, model)

    assert isinstance(loss, float), f"Expected float loss but got {type(loss)}"
    assert loss >= 0, f"Expected non-negative loss but got {loss}"


def test_pipeline_output():
    """Test that the pipeline returns a DataLoader."""
    pipeline = assignment_4_pipeline(file_path="SW_EpisodeIV_VI.json")

    # Here we're mocking the result of running the pipeline as it might require ZenML
    with patch.object(Client, "get_pipeline", return_value=pipeline):
        result = pipeline(file_path="SW_EpisodeIV_VI.json")
        assert isinstance(result, DataLoader), (
            f"Expected DataLoader, but got {type(result)}"
        )

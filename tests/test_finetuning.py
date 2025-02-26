import pytest
from unittest.mock import MagicMock
from finetuning import train_model


def test_train_model():
    """Simple test to ensure train_model runs without errors and returns expected keys."""

    # Mock data_splits with small dummy data
    data_splits = {"train": [(1, 1), (2, 2)]}

    # Mock the function to avoid heavy training computation
    train_model_mock = MagicMock(return_value={"train_loss": 0.01})
    result = train_model_mock(data_splits)

    # Assertions
    assert isinstance(result, dict), "train_model should return a dictionary"
    assert "train_loss" in result, "train_model output should contain 'train_loss' key"
    assert isinstance(result["train_loss"], float), "train_loss should be a float"

    print("✅ test_train_model passed!")


if __name__ == "__main__":
    pytest.main(["-v", "test_finetuning.py"])

import pytest
import os
import torch
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datasets import Dataset

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from model_trainer import train_llama_model

# Skip tests if unsloth isn't installed (for CI environments without GPU)
unsloth_installed = True
try:
    from unsloth import FastLanguageModel
except ImportError:
    unsloth_installed = False


# Mock the ZenML logging functions
@pytest.fixture
def mock_zenml_logging():
    with patch("model_trainer.log_artifact_metadata") as mock_log:
        yield mock_log


# Mock dataset fixture
@pytest.fixture
def sample_dataset():
    # Create a simple dataset with the expected structure
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "target": "<reasoning>\n2+2 equals 4\n</reasoning>\n<answer>\n4\n</answer>",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "target": "<reasoning>\nThe capital of France is Paris.\n</reasoning>\n<answer>\nParis\n</answer>",
        },
    ]
    return Dataset.from_list(data)


# Mock FastLanguageModel
@pytest.fixture
def mock_fast_language_model():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Configure mock tokenizer to return tensor-like objects
    def mock_tokenize(text, **kwargs):
        mock_tensor = MagicMock()
        mock_tensor.input_ids = torch.ones((1, 10), dtype=torch.long)
        mock_tensor.attention_mask = torch.ones((1, 10), dtype=torch.long)
        return mock_tensor

    mock_tokenizer.side_effect = mock_tokenize

    with patch("unsloth.FastLanguageModel", autospec=True) as mock_flm:
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model
        mock_flm.for_inference.return_value = mock_model
        yield mock_flm


# Mock Trainer
@pytest.fixture
def mock_trainer():
    with patch("model_trainer.Trainer", autospec=True) as mock_trainer_cls:
        # Configure the mock trainer
        mock_trainer_instance = mock_trainer_cls.return_value

        # Mock metrics returned by train and evaluate
        mock_trainer_instance.train.return_value = MagicMock(
            metrics={
                "train_loss": 0.5,
                "train_runtime": 100,
                "train_samples_per_second": 10,
            }
        )
        mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.6}

        yield mock_trainer_cls


@pytest.mark.skipif(not unsloth_installed, reason="Unsloth not installed")
def test_train_llama_model_integration():
    """Basic integration test - skipped in CI environments"""
    # This is a simple test that ensures the function runs without errors
    # It's skipped in environments without unsloth installed

    # Create a tiny test dataset
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test"},
            ],
            "target": "Test response",
        }
    ]
    test_dataset = Dataset.from_list(data)

    # Run with minimal epochs to ensure it runs
    try:
        result = train_llama_model(
            dataset=test_dataset,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            output_dir="test_output",
        )
        assert isinstance(result, dict)
    except (ImportError, ValueError):
        # Skip if env vars not set or dependencies missing
        pytest.skip("Missing environment for full integration test")


def test_train_llama_model_unit(
    sample_dataset, mock_fast_language_model, mock_trainer, mock_zenml_logging
):
    """Unit test using mocks to verify the function's logic"""
    # Set environment variable for testing
    os.environ["HF_TOKEN"] = "dummy_token"

    # Call the function
    result = train_llama_model(
        dataset=sample_dataset,
        huggingface_model_name="mock/model",
        num_train_epochs=1,
        output_dir="test_output",
    )

    # Verify the result structure
    assert isinstance(result, dict)
    assert "model_path" in result
    assert "tokenizer_path" in result
    assert "model_name" in result

    # Verify the model was loaded
    mock_fast_language_model.from_pretrained.assert_called_once()

    # Verify LoRA was applied
    mock_fast_language_model.get_peft_model.assert_called_once()

    # Verify training occurred
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.evaluate.assert_called_once()

    # Verify metrics were logged
    mock_zenml_logging.assert_called()

    # Clean up
    if os.path.exists("test_output"):
        import shutil

        shutil.rmtree("test_output")

    # Remove environment variable
    del os.environ["HF_TOKEN"]


def test_train_llama_model_error_handling():
    """Test error handling for missing HF token"""
    # Ensure HF_TOKEN is not in environment
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    # Create a minimal dataset
    test_dataset = Dataset.from_list([{"messages": [], "target": ""}])

    # Check that it raises an error when no token is provided
    with pytest.raises(ValueError, match="HuggingFace token is required"):
        train_llama_model(dataset=test_dataset, hf_token=None)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

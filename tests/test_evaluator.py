import pytest
import os
import torch
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from model_evaluator import (
    extract_structured_answer,
    strict_format_reward_func,
    soft_format_reward_func,
    test_model,
)


@pytest.fixture
def mock_zenml_logging():
    with patch("model_evaluator.log_artifact_metadata") as mock_log:
        yield mock_log


@pytest.fixture
def mock_fast_language_model():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Configure mock tokenizer to return tensor-like objects
    def mock_tokenize(text, **kwargs):
        mock_tensor = MagicMock()
        mock_tensor.input_ids = torch.ones((1, 10), dtype=torch.long)
        return mock_tensor

    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.decode.return_value = (
        "<reasoning>\nTest reasoning\n</reasoning>\n<answer>\nTest answer\n</answer>"
    )

    with patch("unsloth.FastLanguageModel", autospec=True) as mock_flm:
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.for_inference.return_value = mock_model
        yield mock_flm


def test_extract_structured_answer():
    """Test extracting reasoning and answer from structured text"""
    # Test well-formatted response
    text = "<reasoning>\nThis is the reasoning\n</reasoning>\n<answer>\nThis is the answer\n</answer>"
    reasoning, answer = extract_structured_answer(text)
    assert reasoning == "This is the reasoning"
    assert answer == "This is the answer"

    # Test malformed response
    text = "This is just some text without structure"
    reasoning, answer = extract_structured_answer(text)
    assert reasoning == ""
    assert answer == "This is just some text without structure"

    # Test partial structure (only reasoning)
    text = "<reasoning>Only reasoning</reasoning>"
    reasoning, answer = extract_structured_answer(text)
    assert reasoning == "Only reasoning"
    assert answer == ""

    # Test partial structure (only answer)
    text = "<answer>Only answer</answer>"
    reasoning, answer = extract_structured_answer(text)
    assert reasoning == ""
    assert answer == "Only answer"


def test_strict_format_reward_func():
    """Test the strict format reward function"""
    # Correctly formatted text
    text = "<reasoning>\nThis is reasoning\n</reasoning>\n<answer>\nThis is the answer\n</answer>\n"
    assert strict_format_reward_func(text) == 1.0

    # Incorrectly formatted text
    text = "This is not formatted correctly"
    assert strict_format_reward_func(text) == 0.0

    # Missing newlines
    text = "<reasoning>No newlines</reasoning><answer>No newlines</answer>"
    assert strict_format_reward_func(text) == 0.0


def test_soft_format_reward_func():
    """Test the soft format reward function"""
    # Correctly formatted text
    text = "<reasoning>This is reasoning</reasoning><answer>This is the answer</answer>"
    assert soft_format_reward_func(text) == 1.0

    # Incorrectly formatted text
    text = "This is not formatted correctly"
    assert soft_format_reward_func(text) == 0.0

    # With extra whitespace
    text = (
        "<reasoning>This is reasoning</reasoning>  <answer>This is the answer</answer>"
    )
    assert soft_format_reward_func(text) == 1.0


def test_test_model_unit(mock_fast_language_model, mock_zenml_logging):
    """Unit test for the test_model function"""
    # Set environment variable for testing
    os.environ["HF_TOKEN"] = "dummy_token"

    # Configure the mock model to generate text
    mock_model = mock_fast_language_model.from_pretrained.return_value[0]
    mock_model.generate.return_value = torch.ones((1, 20), dtype=torch.long)

    # Create test data
    model_info = {
        "model_path": "test_model_path",
        "tokenizer_path": "test_tokenizer_path",
    }
    test_questions = ["Test question 1", "Test question 2"]

    # Call the function
    result = test_model(
        model_info=model_info, test_questions=test_questions, hf_token="dummy_token"
    )

    # Verify the result structure
    assert isinstance(result, dict)
    assert "questions" in result
    assert "answers" in result
    assert "formatted_qa_pairs" in result
    assert len(result["answers"]) == 2

    # Verify model loading and inference
    mock_fast_language_model.from_pretrained.assert_called_once()
    mock_fast_language_model.for_inference.assert_called_once()

    # Verify generation occurred for both questions
    assert mock_model.generate.call_count == 2

    # Verify metrics were logged
    mock_zenml_logging.assert_called_once()

    # Verify answers.txt was created
    assert os.path.exists("answers.txt")

    # Clean up
    os.remove("answers.txt")

    # Remove environment variable
    del os.environ["HF_TOKEN"]


def test_test_model_error_handling():
    """Test error handling for missing HF token"""
    # Ensure HF_TOKEN is not in environment
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    # Create test data
    model_info = {"model_path": "test_path"}
    test_questions = ["Test question"]

    # Check that it raises an error when no token is provided
    with pytest.raises(ValueError, match="HuggingFace token is required"):
        test_model(model_info=model_info, test_questions=test_questions, hf_token=None)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

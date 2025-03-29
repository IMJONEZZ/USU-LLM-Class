import pytest
import torch
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, LlamaForCausalLM
from evaluator import evaluator  # Modified to return generated_text


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    # Mock the decode method
    tokenizer.decode = MagicMock(return_value="Mock generated text.")

    # Mock the __call__ method (used by tokenizer(prompt, ...))
    def tokenizer_call(prompt, **kwargs):
        return {
            "input_ids": torch.tensor(
                [[0, 1, 2]]
            ),  # Ensure this matches the expected shape
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    tokenizer.__call__ = MagicMock(side_effect=tokenizer_call)
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=LlamaForCausalLM)

    # Add a mock config attribute
    model.config = MagicMock()
    model.config.vocab_size = 32000  # Set vocab_size
    model.config.pad_token_id = None  # Set pad_token_id

    # Mock the generate method
    def generate(input_ids, **kwargs):
        return torch.tensor([[0, 1, 2, 3, 4]])

    model.generate = MagicMock(side_effect=generate)
    return model


def test_tokenizer_decode_called_properly(mock_tokenizer, mock_model):
    with (
        patch("evaluator.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("evaluator.LlamaForCausalLM.from_pretrained", return_value=mock_model),
    ):
        result = evaluator()

        # Verify decode was called correctly
        mock_tokenizer.decode.assert_called_once_with(
            torch.tensor([0, 1, 2, 3, 4]),  # Should match mock_model's output
            skip_special_tokens=True,
        )
        assert result == "Mock generated text."


def test_evaluator_happy_path(mock_tokenizer, mock_model):
    with (
        patch("evaluator.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("evaluator.LlamaForCausalLM.from_pretrained", return_value=mock_model),
        patch("evaluator.torch.cuda.is_available", return_value=True),
    ):
        result = evaluator()

        # Verify model loading
        mock_tokenizer.from_pretrained.assert_called_once_with("star_wars_llama")
        mock_model.from_pretrained.assert_called_once_with("star_wars_llama")

        # Verify tokenizer setup
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
        assert mock_model.config.pad_token_id == mock_tokenizer.pad_token_id

        # Verify device handling
        mock_model.to.assert_called_once_with("cuda")

        # Verify generation arguments
        mock_model.generate.assert_called_once()
        call_args = mock_model.generate.call_args[1]
        assert call_args["max_length"] == 100
        assert call_args["pad_token_id"] == mock_tokenizer.pad_token_id
        assert call_args["attention_mask"] is not None

        # Verify output
        mock_tokenizer.decode.assert_called_once()
        assert result == "Mock generated text."


def test_device_handling(mock_tokenizer, mock_model):
    with (
        patch("evaluator.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("evaluator.LlamaForCausalLM.from_pretrained", return_value=mock_model),
        patch("evaluator.torch.cuda.is_available", return_value=False),
    ):
        evaluator()
        mock_model.to.assert_called_once_with("cpu")


def test_vocab_size_error(mock_tokenizer, mock_model):
    # Create problematic input
    def encode_plus_vocab_error(prompt, **kwargs):
        return {
            "input_ids": torch.tensor([[32001]]),  # Exceeds vocab_size=32000
            "attention_mask": torch.tensor([[1]]),
        }

    mock_tokenizer.encode_plus = MagicMock(side_effect=encode_plus_vocab_error)

    with (
        patch("evaluator.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("evaluator.LlamaForCausalLM.from_pretrained", return_value=mock_model),
    ):
        with pytest.raises(ValueError, match="outside the model's vocabulary"):
            evaluator()


def test_model_loading_error():
    with patch(
        "evaluator.AutoTokenizer.from_pretrained",
        side_effect=OSError("Model not found"),
    ):
        with pytest.raises(OSError, match="Model not found"):
            evaluator()


def test_generation_parameters(mock_tokenizer, mock_model):
    with (
        patch("evaluator.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("evaluator.LlamaForCausalLM.from_pretrained", return_value=mock_model),
    ):
        evaluator()

        call_args = mock_model.generate.call_args[1]
        assert call_args["num_return_sequences"] == 1
        assert call_args["no_repeat_ngram_size"] == 2
        assert call_args["top_k"] == 50
        assert call_args["top_p"] == 0.95
        assert call_args["temperature"] == 0.7


def test_tokenizer_configuration(mock_tokenizer, mock_model):
    with (
        patch("evaluator.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("evaluator.LlamaForCausalLM.from_pretrained", return_value=mock_model),
    ):
        evaluator()

        # Verify tokenizer configuration
        mock_tokenizer.encode_plus.assert_called_once_with(
            "A long time ago in a galaxy far, far away,",
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Verify pad token propagation
        assert mock_model.config.pad_token_id == mock_tokenizer.pad_token_id

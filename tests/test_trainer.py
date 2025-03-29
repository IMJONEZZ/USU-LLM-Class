import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments
from trl import SFTTrainer
from trainer import SFT_train


# Mock the necessary components
@pytest.fixture
def mock_train_dataset():
    class MockDataset:
        def __init__(self):
            self.dataset = [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                },
                {
                    "input_ids": torch.tensor([4, 5, 6]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                },
            ]

    return MockDataset()


@pytest.fixture
def mock_tokenizer():
    return MagicMock(spec=AutoTokenizer)


@pytest.fixture
def mock_model():
    return MagicMock(spec=LlamaForCausalLM)


@pytest.fixture
def mock_training_args():
    return MagicMock(spec=TrainingArguments)


@pytest.fixture
def mock_sft_trainer():
    return MagicMock(spec=SFTTrainer)


def test_SFT_train(
    mock_train_dataset, mock_tokenizer, mock_model, mock_training_args, mock_sft_trainer
):
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ),
        patch("transformers.LlamaForCausalLM.from_pretrained", return_value=mock_model),
        patch("transformers.TrainingArguments", return_value=mock_training_args),
        patch("trl.SFTTrainer", return_value=mock_sft_trainer),
        patch("torch.cuda.empty_cache") as mock_empty_cache,
        patch("torch.cuda.is_available", return_value=True),
    ):
        # Call the function to be tested
        SFT_train(mock_train_dataset)

        # Assertions to ensure the function behaves as expected
        mock_empty_cache.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-1B"
        )
        mock_model.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16,
        )
        mock_model.to.assert_called_once_with("cuda")
        mock_training_args.assert_called_once_with(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=500,
            report_to="none",
        )
        mock_sft_trainer.assert_called_once_with(
            model=mock_model,
            args=mock_training_args,
            train_dataset=mock_train_dataset.dataset,
            data_collator="ANY",  # Use ANY to match the lambda function
        )
        mock_sft_trainer.return_value.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once_with("./star_wars_llama")
        mock_tokenizer.save_pretrained.assert_called_once_with("./star_wars_llama")


# Additional tests can be added to cover edge cases, such as when CUDA is not available
def test_SFT_train_no_cuda(
    mock_train_dataset, mock_tokenizer, mock_model, mock_training_args, mock_sft_trainer
):
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ),
        patch("transformers.LlamaForCausalLM.from_pretrained", return_value=mock_model),
        patch("transformers.TrainingArguments", return_value=mock_training_args),
        patch("trl.SFTTrainer", return_value=mock_sft_trainer),
        patch("torch.cuda.empty_cache") as mock_empty_cache,
        patch("torch.cuda.is_available", return_value=False),
    ):
        # Call the function to be tested
        SFT_train(mock_train_dataset)

        # Assertions to ensure the function behaves as expected
        mock_empty_cache.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-1B"
        )
        mock_model.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16,
        )
        mock_model.to.assert_called_once_with(
            "cpu"
        )  # Should use CPU when CUDA is not available
        mock_training_args.assert_called_once_with(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=500,
            report_to="none",
        )
        mock_sft_trainer.assert_called_once_with(
            model=mock_model,
            args=mock_training_args,
            train_dataset=mock_train_dataset.dataset,
            data_collator="ANY",  # Use ANY to match the lambda function
        )
        mock_sft_trainer.return_value.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once_with("./star_wars_llama")
        mock_tokenizer.save_pretrained.assert_called_once_with("./star_wars_llama")

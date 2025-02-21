# test_trainer.py
import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict
import shutil
from transformers import TrainingArguments


@pytest.fixture
def sample_data():
    return {
        "schema": "User(name, email)",
        "question": "How many users are there?",
        "cypher": "MATCH (u:User) RETURN COUNT(u)",
    }


@pytest.fixture
def small_dataset(sample_data):
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "schema": [sample_data["schema"]] * 2,
                    "question": [sample_data["question"]] * 2,
                    "cypher": [sample_data["cypher"]] * 2,
                }
            )
        }
    )


@pytest.fixture
def processed_small_dataset(small_dataset):
    from data import tokenize_dataset, format_instruction
    from model import initialize_model_and_tokenizer

    model, tokenizer = initialize_model_and_tokenizer()
    small_dataset["train"] = small_dataset["train"].map(format_instruction)
    return tokenize_dataset(small_dataset, tokenizer, max_length=512)


def test_data_loading():
    from data import load_and_preprocess_data

    dataset = load_and_preprocess_data(download_mode="force_redownload")
    assert "train" in dataset
    assert "test" in dataset
    assert "text" in dataset["train"].features


def test_format_instruction(sample_data):
    from data import format_instruction

    result = format_instruction(sample_data)
    text = result["text"]

    assert "<|im_start|>system" in text
    assert sample_data["schema"] in text
    assert "<|im_start|>user" in text
    assert sample_data["question"] in text
    assert "<|im_start|>assistant" in text
    assert sample_data["cypher"] in text


def test_tokenization(processed_small_dataset):
    from model import initialize_model_and_tokenizer

    model, tokenizer = initialize_model_and_tokenizer()

    example = processed_small_dataset["train"][0]
    assert "input_ids" in example
    assert "attention_mask" in example
    assert len(example["input_ids"]) == 512
    assert tokenizer.pad_token_id in example["input_ids"]


@pytest.fixture
def dummy_training_args():
    # Override the training args to run only 1 training step for testing.
    return TrainingArguments(
        output_dir="./temp_test_output",
        max_steps=1,
        per_device_train_batch_size=1,
        evaluation_strategy="no",
        logging_steps=1,
        save_steps=1,
    )


@patch("train.get_training_arguments")
def test_trainer_finetuning(
    mock_get_args, processed_small_dataset, dummy_training_args
):
    # Patch training args for a fast run
    mock_get_args.return_value = dummy_training_args

    from model import initialize_model_and_tokenizer
    from train import initialize_trainer
    import torch

    model, tokenizer = initialize_model_and_tokenizer()
    train_dataset = processed_small_dataset["train"]
    eval_dataset = processed_small_dataset["train"]

    trainer = initialize_trainer(model, tokenizer, train_dataset, eval_dataset)

    # Capture initial parameter states for all trainable parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.clone().detach()

    # Perform a short training run (e.g., one step)
    train_output = trainer.train()
    metrics = train_output.metrics
    assert "train_loss" in metrics, "Expected loss metric not found"

    # Verify that at least one parameter changed during training
    any_changed = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.allclose(initial_params[name], param, atol=1e-6):
                any_changed = True
                break

    assert any_changed, "No parameters changed during training."


@pytest.fixture
def save_dir(tmp_path):
    # Create a subdirectory for saving
    dir_path = tmp_path / "test_save"
    yield dir_path
    # Cleanup: remove the directory and its contents
    shutil.rmtree(dir_path, ignore_errors=True)


def test_model_saving(save_dir):
    from model import initialize_model_and_tokenizer

    model, tokenizer = initialize_model_and_tokenizer()

    # Test saving: files will be saved to save_dir
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Verify files were created
    assert (save_dir / "adapter_config.json").exists()
    assert (save_dir / "adapter_model.safetensors").exists()
    assert (save_dir / "tokenizer.json").exists()

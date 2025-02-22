import pytest
import pandas as pd
import torch
from transformers import BertGenerationConfig
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_trainer import StarWarsDataset, bert_trainer


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {"tokens": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]}
    )


@pytest.fixture
def sample_tokenizer_config():
    return {
        "str_to_int": {
            "<|pad|>": 0,
            "<|endoftext|>": 1,
            "<|unk|>": 2,
            "hello": 3,
            "world": 4,
        }
    }


def test_star_wars_dataset_initialization(sample_dataframe):
    # Test dataset creation
    dataset = StarWarsDataset(sample_dataframe, max_length=10)
    assert len(dataset) == 3

    # Test max_length enforcement
    item = dataset[0]
    assert item["input_ids"].shape[0] == 10
    assert item["attention_mask"].shape[0] == 10


def test_star_wars_dataset_padding(sample_dataframe):
    # Test padding behavior
    dataset = StarWarsDataset(sample_dataframe, max_length=8)
    item = dataset[0]

    # Check if padding is applied correctly
    input_ids = item["input_ids"]
    attention_mask = item["attention_mask"]

    # First 5 should be original tokens, rest should be padding
    assert torch.all(input_ids[5:] == 0)
    assert torch.all(attention_mask[:5] == 1)
    assert torch.all(attention_mask[5:] == 0)


def test_star_wars_dataset_invalid_data():
    # Test handling of invalid data
    invalid_df = pd.DataFrame({"tokens": ["invalid", None, [1, 2, 3]]})
    dataset = StarWarsDataset(invalid_df, max_length=5)
    # Should only keep the valid sequence
    assert len(dataset) == 1


def test_star_wars_dataset_attention_mask():
    # Test attention mask generation
    df = pd.DataFrame({"tokens": [[1, 2, 3]]})
    dataset = StarWarsDataset(df, max_length=5)
    item = dataset[0]

    expected_attention = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long)
    assert torch.all(item["attention_mask"] == expected_attention)


def test_bert_trainer_output_format(sample_dataframe, sample_tokenizer_config):
    """Test the structure of trainer output"""
    result = bert_trainer(
        train_data=sample_dataframe,
        validation_data=sample_dataframe,
        tokenizer_config=sample_tokenizer_config,
        batch_size=2,
        max_length=10,
        num_epochs=1,
    )

    # Check if the output has all required keys
    assert isinstance(result, dict)
    assert all(key in result for key in ["model_state", "config", "training_stats"])

    # Check training stats format
    assert isinstance(result["training_stats"], list)
    if result["training_stats"]:
        stats = result["training_stats"][0]
        assert all(key in stats for key in ["epoch", "train_loss", "val_loss"])


def test_bert_trainer_config_generation(sample_dataframe, sample_tokenizer_config):
    """Test that the trainer creates proper BERT config"""
    result = bert_trainer(
        train_data=sample_dataframe,
        validation_data=sample_dataframe,
        tokenizer_config=sample_tokenizer_config,
        batch_size=2,
        max_length=10,
        num_epochs=1,
    )

    config_dict = result["config"]
    config = BertGenerationConfig(**config_dict)

    # Check essential config parameters
    assert config.vocab_size == len(sample_tokenizer_config["str_to_int"])
    assert config.is_decoder is True
    assert config.add_cross_attention is True


def test_bert_trainer_small_batch(sample_dataframe, sample_tokenizer_config):
    """Test trainer with minimal data to verify basic functionality"""
    small_df = pd.DataFrame({"tokens": [[1, 2, 3]]})

    result = bert_trainer(
        train_data=small_df,
        validation_data=small_df,
        tokenizer_config=sample_tokenizer_config,
        batch_size=1,
        max_length=5,
        num_epochs=1,
    )

    # Verify training completed and produced stats
    assert len(result["training_stats"]) == 1
    assert isinstance(result["training_stats"][0]["train_loss"], float)
    assert isinstance(result["training_stats"][0]["val_loss"], float)


if __name__ == "__main__":
    pytest.main([__file__])

import pytest
import torch
from unittest.mock import patch
from data_loader import load_data
from tokenizer import encode_text
from dataset import NextTokenDataset, split_data, collate_fn
from model import SimpleLanguageModel
from train import train_model
from transformers import BertTokenizerFast

# Load BERT tokenizer (for testing encode_text)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Sample text for testing
SAMPLE_TEXT = "Hello world. This is a test. ZenML is great!"
TOKENIZED_TEXT = [
    [101, 7592, 2088, 102],
    [101, 2023, 2003, 1037, 3231, 102],
    [101, 10924, 2638, 2003, 2307, 102],
]


# ------------------ TEST DATA LOADER ------------------


@patch("builtins.open", create=True)
def test_load_data(mock_open):
    """Test that load_data correctly reads and processes JSON."""
    mock_open.return_value.__enter__.return_value.read.return_value = (
        '[{"Line": "Hello world."}, {"Line": "ZenML is great!"}]'
    )

    result = load_data()
    assert result == "Hello world. ZenML is great!", (
        "load_data() output does not match expected text"
    )


# ------------------ TEST TOKENIZATION ------------------


def test_encode_text():
    """Test that encode_text correctly tokenizes text."""
    result = encode_text(SAMPLE_TEXT)
    assert isinstance(result, list), "Output should be a list of tokenized sentences"
    assert all(isinstance(seq, list) for seq in result), (
        "Each element should be a list of token IDs"
    )
    assert result == TOKENIZED_TEXT, (
        "Tokenized output does not match expected BERT token IDs"
    )


# ------------------ TEST DATA SPLITTING ------------------


def test_split_data():
    """Test that split_data correctly splits the dataset."""
    sample_sequences = [list(range(5))] * 10  # 10 sequences

    result = split_data(sample_sequences)

    assert "train" in result and "val" in result and "test" in result, (
        "Missing keys in split data"
    )
    assert len(result["train"]) > len(result["val"]) > len(result["test"]), (
        "Incorrect data split ratios"
    )


# ------------------ TEST DATASET CLASS ------------------


def test_next_token_dataset():
    """Test that NextTokenDataset correctly prepares input-output pairs."""
    sample_sequences = [[10, 20, 30, 40], [50, 60, 70]]

    dataset = NextTokenDataset(sample_sequences)

    assert len(dataset) == 2, "Dataset should have 2 sequences"

    x0, y0 = dataset[0]
    assert list(x0.numpy()) == [10, 20, 30], "Incorrect x values"
    assert list(y0.numpy()) == [20, 30, 40], "Incorrect y values"


# ------------------ TEST COLLATE FUNCTION ------------------


def test_collate_fn():
    """Test that collate_fn correctly pads sequences."""
    batch = [
        (torch.tensor([1, 2]), torch.tensor([2, 3])),
        (torch.tensor([4]), torch.tensor([5])),
    ]

    padded_x, padded_y = collate_fn(batch, max_len=4)

    assert padded_x.shape == (2, 4), "Incorrect padding for x"
    assert padded_y.shape == (2, 4), "Incorrect padding for y"
    assert padded_x[0, 2] == 0, "Padding value should be 0"
    assert padded_y[1, 1] == -100, "Padding for y should be -100"


# ------------------ TEST MODEL ------------------


def test_model_forward_pass():
    """Test that SimpleLanguageModel produces correct output shape."""
    vocab_size = 30522  # BERT vocab size
    model = SimpleLanguageModel(vocab_size)

    x = torch.randint(0, vocab_size, (2, 5))  # batch_size=2, seq_len=5
    logits = model(x)

    assert logits.shape == (2, 5, vocab_size), "Incorrect model output shape"


# ------------------ TEST TRAINING FUNCTION ------------------


@patch("torch.optim.AdamW.step")
@patch("torch.optim.AdamW.zero_grad")
@patch("torch.optim.AdamW.backward")
def test_train_model(mock_step, mock_zero_grad, mock_backward):
    """Test that train_model runs without errors using a small dataset."""

    # Mock small dataset
    data_splits = {
        "train": [[101, 7592, 2088, 102]],
        "val": [[101, 2023, 2003, 1037, 3231, 102]],
        "test": [[101, 10924, 2638, 2003, 2307, 102]],
    }

    result = train_model(data_splits=data_splits, batch_size=2, epochs=1)

    assert "train_loss" in result, (
        "train_model should return a dictionary with train_loss"
    )
    assert isinstance(result["train_loss"], float), "train_loss should be a float"


# ------------------ MAIN TEST RUN ------------------

if __name__ == "__main__":
    pytest.main()

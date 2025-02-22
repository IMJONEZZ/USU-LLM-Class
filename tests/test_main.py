import pytest
from data_loader import load_data
from tokenizer import encode_text
from dataset import split_data
from train import train_model

### Sample Test Data ###
SAMPLE_TEXT = "Hello world. ZenML is great!"
SAMPLE_TOKENS = [
    [101, 7592, 2088, 102],
    [101, 10924, 2638, 2003, 2307, 102],
]  # Example token IDs
SAMPLE_SPLIT = {
    "train": SAMPLE_TOKENS[:1],
    "val": SAMPLE_TOKENS[:1],
    "test": SAMPLE_TOKENS[:1],
}


### 1️⃣ Load Data ###
def test_load_data():
    """Test if load_data returns a non-empty string."""
    text = load_data()
    assert isinstance(text, str), "load_data should return a string"
    assert len(text) > 0, "load_data should return non-empty text"


### 2️⃣ Tokenization ###
def test_encode_text():
    """Test if encode_text tokenizes text into a list of lists."""
    result = encode_text(SAMPLE_TEXT)
    assert isinstance(result, list), "encode_text should return a list"
    assert all(isinstance(seq, list) for seq in result), (
        "Each tokenized sentence should be a list"
    )
    assert len(result) > 0, "encode_text should not return an empty list"


### 3️⃣ Data Splitting ###
def test_split_data():
    """Test if split_data correctly splits data into train, val, test."""
    result = split_data(SAMPLE_TOKENS)
    assert isinstance(result, dict), "split_data should return a dictionary"
    assert all(key in result for key in ["train", "val", "test"]), (
        "split_data should return train, val, and test keys"
    )
    assert sum(len(v) for v in result.values()) == len(SAMPLE_TOKENS), (
        "Total split data should match input size"
    )


### 4️⃣ Model Training ###
def test_train_model():
    """Test if train_model runs without errors."""
    result = train_model(SAMPLE_SPLIT, batch_size=1, epochs=1)
    assert isinstance(result, dict), "train_model should return a dictionary"
    assert "train_loss" in result, "train_model should return train_loss"

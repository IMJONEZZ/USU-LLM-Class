# bpe_pipeline.py
from zenml import pipeline, step
import json
from Tokenizer_BPE import BPE  # Import the BPE tokenizer

@step
def load_data(file_path: str) -> dict:
    """Loads dialogue lines from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [entry["Line"] for entry in data if "Line" in entry]
    return {"texts": lines}

@step
def train_bpe(data: dict, vocab_size: int) -> dict:
    """Trains a BPE tokenizer."""
    bpe = BPE(data["texts"], vocab_size)
    merges = bpe.train()
    return {"bpe": bpe, "merges": merges}

@step
def tokenize_data(data: dict, bpe_data: dict) -> dict:
    """Tokenizes text data using the trained BPE tokenizer."""
    bpe = bpe_data["bpe"]
    tokenized_texts = [bpe.tokenize(text) for text in data["texts"]]
    return {"tokenized_texts": tokenized_texts}

@pipeline
def bpe_tokenization_pipeline(file_path: str, vocab_size: int):
    """Pipeline for BPE tokenization."""
    data = load_data(file_path)
    bpe_data = train_bpe(data, vocab_size)
    tokenized_data = tokenize_data(data, bpe_data)

if __name__ == "__main__":
    run = bpe_tokenization_pipeline("SW_EpisodeIV_VI.json", vocab_size=100)

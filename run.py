from zenml import pipeline, step
import json
import re
import pandas as pd
from typing import List
from collections import Counter, defaultdict
from BPEtokenizer import BPETokenizer


@step
def load_data() -> dict:
    """Simulates loading of training data and labels."""

    training_data = [[1, 2], [3, 4], [5, 6]]

    labels = [0, 1, 0]

    return {"features": training_data, "labels": labels}
@step
def dummy_step():
    hi = "bye"
    

@step
def train_tokenizer(data: List[str]) -> BPETokenizer:
    "trains the tokenizer on data provided"

    model = BPETokenizer(vocab_size=1000)
    model.build_vocab(data)
    return model

    total_labels = sum(data["labels"])


@pipeline
def llm_pipeline():
    """Define a pipeline that connects the steps."""
    x = dummy_step()

"""
ADD LOGIC FOR DATA LOADER HERE

"""


if __name__ == "__main__":
    run = llm_pipeline()

    # You can now use the `run` object to see steps, outputs, etc.

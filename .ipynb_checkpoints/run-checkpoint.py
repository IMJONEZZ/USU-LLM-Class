from zenml import pipeline, step
import json
import re
import pandas as pd
from typing import List
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from BPEtokenizer import BPETokenizer
from DataLoader import DataLoader


@step

def load_data(data: List[str]) -> DataLoader:

    data = CustomDataset(data)
    return data



@step
def dummy_step():
    hi = "bye"
    

@step
def train_tokenizer(data: list()) -> BPETokenizer:
    "trains the tokenizer on data provided"

    model = BPETokenizer(vocab_size=1000)
    model.build_vocab(data)
    return model

    total_labels = sum(data["labels"])


@pipeline
def llm_pipeline():
    """Define a pipeline that connects the steps."""
    x = dummy_step()
    
    
    #Read in
    with open('SW_EpisodeIV_VI.json', 'r') as file:
        raw_data = json.load(file) 
    if isinstance(raw_data, dict):  # If JSON is a dictionary, extract string values
        data = [str(value) for value in raw_data.values()]
    elif isinstance(raw_data, list):  # If JSON is a list, convert all items to strings
        data = [str(item) for item in raw_data]
    else:
        raise ValueError("Unexpected JSON format: Expected a dictionary or list.")
    dataset = load_data(data)
    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = train_tokenizer(data)



if __name__ == "__main__":
    run = llm_pipeline()

    # You can now use the `run` object to see steps, outputs, etc.

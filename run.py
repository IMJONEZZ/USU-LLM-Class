from zenml import pipeline, step
import json
import re
import pandas as pd
from typing import List
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from BPEtokenizer import BPETokenizer
from DataLoader import CustomDataset
from data import StarWarsDataset


@step
def load_data(file_path: str) -> DataLoader:

    data = StarWarsDataset(file_path)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    return dataloader


@pipeline
def llm_pipeline():
    """Define a pipeline that connects the steps."""
    #Read in
    file_path = 'SW_EpisodeIV_VI.json'
    load_data(file_path)
    # Initialize DataLoader
    



if __name__ == "__main__":
    run = llm_pipeline()

    # You can now use the `run` object to see steps, outputs, etc.

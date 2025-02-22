from zenml import pipeline
from data_loader import load_data
from tokenizer import encode_text
from dataset import split_data
from train import train_model


@pipeline
def train_eval_pipeline():
    """Pipeline for training the LLM model."""
    text = load_data()
    token_ids = encode_text(text=text)
    data_splits = split_data(tokenized_sequences=token_ids)
    train_model(data_splits=data_splits)

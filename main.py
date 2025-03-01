from zenml import pipeline, step
import json
from tokenizer import tokenize_text
from dataloader import create_dataloader
# from trainer import SFT_train
# from evaluator import evaluator


@step
def load_data(file_path: str):
    """Loads JSON data from a file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


@pipeline
def assignment_6_pipeline(file_path: str):
    data = load_data(file_path)
    encoding = tokenize_text(data)
    dataloader = create_dataloader(encoding)
    # trained = SFT_train(dataloader)
    # generated_text = evaluator()
    return dataloader


if __name__ == "__main__":
    assignment_6_pipeline(file_path="SW_EpisodeIV_VI.json")

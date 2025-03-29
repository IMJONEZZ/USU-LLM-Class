from zenml import pipeline, step
import json
from tokenizer import tokenize_text
from dataloader import create_dataloader
# from vectordb import vectordb, get_embeddings
from html_output import html_outlines_output
# from trainer import SFT_train
# from evaluator import evaluator


@step
def load_data(file_path: str):
    """Loads JSON data from a file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


@step
def load_query_data():
    """Loads (or defines) query data."""
    # Here, we hardcode a simple query.
    return [{"Line": "Hello"}]


@pipeline
def assignment_9_pipeline(file_path: str):
    data = load_data(file_path)
    encoding = tokenize_text(data)
    dataloader = create_dataloader(encoding)
    # dataset_embeddings = get_embeddings(encoding)
    # query_data = load_query_data()
    # query_encoding = tokenize_text(query_data)
    # query_embeddings = get_embeddings(query_encoding)
    # vectordb(dataset_embeddings, query_embeddings)
    # trained = SFT_train(dataloader)
    # generated_text = evaluator()
    html_output = html_outlines_output()
    return html_output


if __name__ == "__main__":
    assignment_9_pipeline(file_path="SW_EpisodeIV_VI.json")

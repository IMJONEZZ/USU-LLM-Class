from zenml import pipeline, step
from tokenizer import WordPieceTokenizer
import json


# @step
def load_data() -> dict:
    with open('SW_EpisodeIV_VI.json', 'r') as file:
        data: dict = json.load(file)

    return data


# @pipeline
def simple_ml_pipeline():

   """Define a pipeline that connects the steps."""

   dataset = load_data()
   tokenizer = WordPieceTokenizer(dataset['vocab'])

 

if __name__ == "__main__":

#    run = simple_ml_pipeline()
    data = load_data()
    print(data[1])
   # You can now use the `run` object to see steps, outputs, etc.
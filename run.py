# from zenml import pipeline, step
from tokenizer import WordPieceTokenizer
import json
import re

# @step
def load_data() -> list:
    with open('SW_EpisodeIV_VI.json', 'r') as file:
        data = json.load(file)
    corpus = [entry["Line"] for entry in data]
    corpus = [re.split(r'([,.:;?_!"()\']|--|\s)', line) for line in corpus]
    corpus = [word.lower() for sentence in corpus for word in sentence if word != " " and word != ""]

    return corpus


# @pipeline
def simple_ml_pipeline():

   """Define a pipeline that connects the steps."""

   dataset = load_data()
   tokenizer = WordPieceTokenizer(dataset['vocab'])

 

if __name__ == "__main__":

#    run = simple_ml_pipeline()
    data = load_data()
    # print(data)
    tokenizer = WordPieceTokenizer(vocab_size=1000, corpus=data)
    # tokenizer.build_vocab(data)
   # You can now use the `run` object to see steps, outputs, etc.
# from zenml import pipeline, step
from tokenizer import WordPieceTokenizer
import json


# @step
def load_data() -> str:
    with open('SW_EpisodeIV_VI.json', 'r') as file:
        raw_string = ""
        # data: dict = json.load(file)
        # for j_object in data:
        #     raw_string += j_object['Line'] + " "
        for line in file:
            if "Line" in line:
                raw_string += line.split(":")[1].strip().strip('"') + " "
                # print(line.split(":")[1].strip().strip('"')) 
    return raw_string


# @pipeline
def simple_ml_pipeline():

   """Define a pipeline that connects the steps."""

   dataset = load_data()
   tokenizer = WordPieceTokenizer(dataset['vocab'])

 

if __name__ == "__main__":

#    run = simple_ml_pipeline()
    data = load_data()
    print(data)
   # You can now use the `run` object to see steps, outputs, etc.
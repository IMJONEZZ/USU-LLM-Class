from zenml import pipeline, step
from datasets import load_dataset

class SimpleTokenizer: 
    def __init__(self, vocab):
       self.str_to_int = vocab
       self.int_to_str = {v: k for k, v in vocab.items()}
   
    def encode(self, text):
       preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
       preprocessed = [
           item.strip() for item in preprocessed if item.strip()
       ]
       preprocessed = [item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed]
       ids = [self.str_to_int[s] for s in preprocessed]
       return ids
   
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text

@step

def load_data() -> dict:
    ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    return ds


@step

def train_model(data: dict) -> None:

   """

   A mock 'training' process that also demonstrates using the input data.

   In a real-world scenario, this would be replaced with actual model fitting logic.

   """

   total_features = sum(map(sum, data['features']))

   total_labels = sum(data['labels'])

   print(f"Trained model using {len(data['features'])} data points. "

         f"Feature sum is {total_features}, label sum is {total_labels}")

 

@pipeline

def simple_ml_pipeline():

   """Define a pipeline that connects the steps."""

   dataset = load_data()
   train_model(dataset)

 

if __name__ == "__main__":

   run = simple_ml_pipeline()

   # You can now use the `run` object to see steps, outputs, etc.
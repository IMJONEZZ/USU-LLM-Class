import re
from zenml import pipeline, step
from datasets import load_dataset
from collections import Counter

#Tokenizer class
class ImprovedTokenizer:
   def __init__(self, vocab):
      self.str_to_int = vocab
      self.int_to_str = { i:s for s,i in vocab.items()}

   '''this new preprocess function will allow me to improve our original tokenizer by making the
   text lowercase and dealing with contractions like "they've" '''
   def preprocess(self, text):

      '''by turning all words lowercase, I prevent words from being counted twice.
      ex: (idk if this is actually in the dataset) "Lightsaber" and "lightsaber" to just "lightsaber" '''
      text = text.lower()


      '''now I will get rid of contractions, which will presumably make the LLM better? Instead of
      having to deal with"they've" or "he'll" we can just replace those with "they have" or "he will" '''

      #dictionary of most likely contractions:
      contraction_dict = {
         'm': 'am',
         's': 'is',
         't': 'not',
         're': 'are',
         've': 'have',
         'll': 'will',
         'd': 'would',
         'N': 'not',
         'n': 'not',
         'c': 'can'
      }

      '''I learned while working on this very project that we can use the .sub method to divide a word
      like "they've" into two groups: the one before the apostrophe and the one after it. I intend
      to replace the second group with the values in our contraction_dict (like "m" to "am") '''

      def replace_contractions(match):
         #word before the apostrophe
         word = match.group(1)
         #contraction that follows apostrophe
         contraction = match.group(2)

         #.get method on our contraction_dict finds the expanded form of any matched contractions
         replaced = contraction_dict.get(contraction, contraction)
         #return the replaced or expanded form
         return f"{word} {replaced}"

      text = re.sub(r"([a-zA-Z])'(m|s|t|re|ve|ll|d|N|n|c)(?=\s|$)", replace_contractions, text)
      return text

   def encode(self, text):
      #use our preprocess function for the text
      text = self.preprocess(text)

      #split the text into tokens
      preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
      preprocessed = [
         item.strip() for item in preprocessed if item.strip()
      ]
      preprocessed = [item if item in self.str_to_int
                      else "<|unk|>" for item in preprocessed]

      #encode the tokens into ids
      ids = [self.str_to_int[s] for s in preprocessed]
      return ids

   def decode(self, ids):
      text = " ".join([self.int_to_str[i] for i in ids])
      text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
      return text
       
@step

def load_and_tokenize_data() -> dict:

   """Simulates loading of training data and labels."""

   #import dataset
   dataset = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI", split="train[:10%]")

   #make a usable data structure from our dataset
   vocab_list = []
   for line in dataset["Line"]:
      vocab_list.extend(re.split(r'([,.:;?_!"()\']|--|\s)', line))

   #counts word frequencies using Counter, which I learned about while working on this assignment
   word_counts = Counter(vocab_list)

   '''I changed the terms that are in the parenthesis because for our new
   dataset, we are dealing with lines and characters '''
   training_data = dataset["Line"]

   labels = dataset["Character"]

   #define a vocab dictionary for our tokenizer to use
   vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}

   #set up tokenizer object
   tokenizer = ImprovedTokenizer(vocab)

   #print out our first line of the dataset
   first_line = dataset["Line"][0]
   print(f"orig: {first_line}")

   #encode first line
   encoded_first_line = tokenizer.encode(first_line)
   print(f"endcoded: {encoded_first_line}")

   #decode encoded line
   decoded_first_line = tokenizer.decode(encoded_first_line)
   print(f"decoded: {decoded_first_line}")

   print("hope this works")
   #this was essentiallly here to begin with so i left it untouched
   return {'features': training_data, 'labels': labels}

@step

def train_model(data: dict) -> None:

   #I essentially left this untouched from our last assignment

   total_features = len(data['features'])

   total_labels = len(data['labels'])

   print(f"Trained model using {total_features} data points. label sum is {total_labels}")


#define the pipeline
@pipeline

#i changed the function name because this is a new and improved pipeline!
def my_pipeline_with_tokenization():

   """Define a pipeline that connects the steps."""

   dataset = load_and_tokenize_data()
   train_model(dataset)

if __name__ == "__main__":

   run = my_pipeline_with_tokenization()


   # You can now use the `run` object to see steps, outputs, etc.


from zenml import pipeline, step
import json
import re
import collections



class SimpleTokenizer: 
   def __init__(self, vocab): 
      self.str_to_int = vocab[0]
      self.str_to_freq = vocab[1]
      self.int_to_str = { i:s for s,i in vocab[0].items()}

   def encode(self, text): 
      preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
      preprocessed = [ 
            item.strip() for item in preprocessed if item.strip() 
      ] 
      pairs = []
      for word in preprocessed:
         word = list(word)
         if len(word) > 1:
            count = 0
            while count < len(word) - 1:
               pairs.append(word[count] + word[count+1])
               count += 1
      pairs1 = [item if item in self.str_to_int  
            else "<|unk|>" for item in pairs] 
      ids = [self.str_to_int[s] for s in pairs1]
      pairs2 = [item if item in self.str_to_freq  
            else "<|unk|>" for item in pairs] 
      freqs = [self.str_to_freq[s] for s in pairs2]
      return [ids, freqs]

   def decode(self, ids): 
      text = " ".join([self.int_to_str[i] for i in ids]) 
      text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) 
      return text
 


@step

def tokenizer(vocab: list, text: str) -> list:
   tokenizer = SimpleTokenizer(vocab)
   return tokenizer.encode(text)



@step

def build_vocab() -> list:
   # Load in file 
   with open('SW_EpisodeIV_VI.json', 'r') as file:
      data = json.load(file)

   # Create array of words
   preprocessed = []
   for i in range(0, len(data) - 1):
      split_text = re.split(r'([,.:;?_!"()\']|--|\s)', data[i]['Line'])
      split_text = [ 
         item.strip() for item in split_text if item.strip() 
      ] 
      preprocessed.extend(split_text) 

   # Build Vocab 
   all_tokens = sorted(list(set(preprocessed))) 
   pairs = []
   for word in all_tokens:
      word = list(word)
      if len(word) > 1:
         count = 0
         while count < len(word) - 1:
            pairs.append(word[count] + word[count+1])
            count += 1
   pairs.append("<|unk|>")
   freqs = collections.Counter(pairs)
   pairs = sorted(list(set(pairs)))
   vocab = [
      {token:integer for integer,token in enumerate(pairs)},
      freqs
   ]
   return vocab



@pipeline

def assignment_2_pipeline():

   """Define a pipeline that connects the steps."""

   dataset = build_vocab()

   tokenizer(dataset, "Star Wars Palpatine Skywalker")



if __name__ == "__main__":

   run = assignment_2_pipeline()

   # You can now use the `run` object to see steps, outputs, etc.
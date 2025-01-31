from zenml import pipeline, step
<<<<<<< Updated upstream
<<<<<<< Updated upstream

import pandas as pd 
import json
import re
from collections import Counter, defaultdict

df = pd.read_json("hf://datasets/andrewkroening/Star-wars-scripts-dialogue-IV-VI/SW_EpisodeIV_VI.json")

@step

def bpe_vocab_step(lines: list, vocab_size: int = 1000) -> dict:
    """
    Build a Byte Pair Encoding (BPE) vocabulary.

    Args:
        lines (list): List of text lines to build the vocabulary.
        vocab_size (int): The maximum size of the vocabulary.

    Returns:
        dict: A dictionary with the BPE vocabulary and token mappings.
    """
    # BPEVocab logic

    class BPEVocab:
        def __init__(self, lines, vocab_size=1000):
            self.vocab_size = vocab_size
            self.tokens = ["<|endoftext|>", "<|unk|>"]
            self.bpe_vocab = defaultdict(int)
            self.build_vocab(lines)
    
        def build_vocab(self, lines):
            # Tokenize each line into characters initially
            word_freq = Counter(lines)
            for word, freq in word_freq.items():
                tokens = list(word) + ["</w>"]
                self.bpe_vocab[tuple(tokens)] = freq
    
            # Iteratively merge tokens to form subword units
            for _ in range(self.vocab_size - len(self.tokens)):
                pairs = self.get_stats()
                if not pairs:
                    break
                best_pair = max(pairs, key=pairs.get)
                self.merge_pair(best_pair)
    
            # Create token-integer mapping
            self.str_to_int = {token: i for i, token in enumerate(self.tokens)}
            self.int_to_str = {i: token for token, i in self.str_to_int.items()}
    
        def get_stats(self):
            pairs = Counter()
            for token_list, freq in self.bpe_vocab.items():
                for i in range(len(token_list) - 1):
                    pairs[(token_list[i], token_list[i + 1])] += freq
            return pairs
    
        def merge_pair(self, pair):
            new_vocab = defaultdict(int)
            bigram = " ".join(pair)
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            for token_list, freq in self.bpe_vocab.items():
                new_token_list = tuple(re.sub(pattern, "".join(pair), " ".join(token_list)).split())
                new_vocab[new_token_list] += freq
            self.bpe_vocab = new_vocab
            self.tokens.append("".join(pair))
    
    # Tokenizer with BPE
    class BPETokenizer:
        def __init__(self, bpe_vocab):
            self.str_to_int = bpe_vocab.str_to_int
            self.int_to_str = bpe_vocab.int_to_str
    
        def encode(self, text):
            tokens = list(text) + ["</w>"]
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                pair_indices = [(self.str_to_int.get(p[0]), self.str_to_int.get(p[1])) for p in pairs]
                best_pair = min(pair_indices, default=None, key=lambda x: x if None not in x else float('inf'))
                if best_pair is None:
                    break
                tokens = self.merge_tokens(tokens, best_pair)
    
            token_ids = [self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in tokens]
            return token_ids
        def merge_tokens(self, tokens, pair):
            """Merges the best pair of tokens."""
            merged = []
            skip = False
            for i in range(len(tokens) - 1):
                if skip:
                    skip = False
                    continue
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    merged.append(pair[0] + pair[1])  # Merge the pair
                    skip = True  # Skip the next token as it's merged
                else:
                    merged.append(tokens[i])
            if not skip:  # Add the last token if it wasn't merged
                merged.append(tokens[-1])
            return merged

        def decode(self, token_ids):
            """Decodes a sequence of token IDs back into a string."""
            tokens = [self.int_to_str[token_id] for token_id in token_ids]
            # Join tokens and remove special subword indicators like "</w>"
            text = "".join(tokens).replace("</w>", " ").strip()
            return text

@step

def train_model(data) -> None:

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

   bpe_vocabulary = bpe_vocab_step(df.to_json())
 

if __name__ == "__main__":

   run = simple_ml_pipeline()

   # You can now use the `run` object to see steps, outputs, etc.
=======
=======
>>>>>>> Stashed changes
import json
import re
import pandas as pd
from typing import List
from collections import Counter, defaultdict
from BPEtokenizer import BPETokenizer


@step
def load_data() -> dict:
    """Simulates loading of training data and labels."""

    training_data = [[1, 2], [3, 4], [5, 6]]

    labels = [0, 1, 0]

    return {"features": training_data, "labels": labels}
@step
def dummy_step():
    hi = "bye"
    

@step
def train_tokenizer(data: List[str]) -> BPETokenizer:
    "trains the tokenizer on data provided"

    model = BPETokenizer(vocab_size=1000)
    model.build_vocab(data)
    return model

    total_labels = sum(data["labels"])


@pipeline
def llm_pipeline():
    """Define a pipeline that connects the steps."""
    x = dummy_step()

"""
ADD LOGIC FOR DATA LOADER HERE

"""


if __name__ == "__main__":
    run = llm_pipeline()

    # You can now use the `run` object to see steps, outputs, etc.
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

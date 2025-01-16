import json
import re

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.cache = {}

    def encode(self, text):
        """Encode the input text to a list of token IDs."""
        preprocessed = re.findall(r"\.\.\.|[\w']+|[.,!?;()\"]", text)
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """Decode a list of token IDs back to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
with open('starWars.json', 'r') as file:
    data = json.load(file)
texts = [item["Line"] for item in data]

preprocessed = [word for text in texts for word in re.findall(r"\.\.\.|[\w']+|[.,!?;()\"]", text)]
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

tokenizer = SimpleTokenizer(vocab)

for text in texts:
    print(f"Original Text: {text}")
    # Encode
    token_ids = tokenizer.encode(text)
    print(f"Encoded Tokens: {token_ids}")
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded Text: {decoded_text}")
    print("---")
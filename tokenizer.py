import re
from collections import defaultdict, Counter

# Tokenize
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [ item.strip() for item in preprocessed if item.strip() ]

        preprocessed = [item if item in self.str_to_int
        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class WordPieceTokenizer:
    def __init__(self, vocab_size: int, corpus: list):
        self.vocab_size = vocab_size
        self.str_to_int = self.build_vocab(corpus)
        # self.int_to_str = { i:s for s,i in self.str_to_int.items()}

    def encode(self, text):
        
        return ""

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
    def build_vocab(self, corpus: list) -> dict:
        vocab = {char: count for char, count in Counter("".join(corpus)).items()}
        print(vocab)
        
        while len(vocab) < self.vocab_size:
            # Step 2: Find most frequent subword pair
            pair_freqs = self._calculate_pair_frequencies(corpus)
            if not pair_freqs:
                break
            
            # Merge the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = "".join(best_pair)
            vocab[new_token] = pair_freqs[best_pair]
            
            # Update corpus with merged tokens
            corpus = [word.replace(" ".join(best_pair), new_token) for word in corpus]
        print(vocab)
        return vocab
    
    def _calculate_pair_frequencies(self, corpus):
        """Calculate frequencies of adjacent pairs in the corpus."""
        pair_freqs = defaultdict(int)
        for word in corpus:
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += 1
        return pair_freqs
    
if __name__ == "__main__":
    print("Wrong file")
    # tokenizer = WordPieceTokenizer(vocab)
    # text = "Hello, World."
    # ids = tokenizer.encode(text)
    # print(ids)
    # text = tokenizer.decode(ids)
    # print(text)
    # print(tokenizer.int_to_str.keys())
    # print(tokenizer.int_to_str.values())
import re
import string

# Get all words
full_document_text = "Hello, World.<|endoftext|> This is a test.<|endoftext|>"
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', full_document_text)
preprocessed = [
    item.strip() for item in preprocessed if item.strip()
]

# Build Vocab
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))

# Tokenize
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

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
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class WordPieceTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        
        return ids

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
if __name__ == "__main__":
    tokenizer = WordPieceTokenizer(vocab)
    text = "Hello, World."
    ids = tokenizer.encode(text)
    print(ids)
    text = tokenizer.decode(ids)
    print(text)
    print(tokenizer.int_to_str.keys())
    print(tokenizer.int_to_str.values())
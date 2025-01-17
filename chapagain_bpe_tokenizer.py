from zenml.pipelines import pipeline
from zenml.steps import step
from collections import Counter, defaultdict
import json
import re


class BPETokenizer:
    """A simple implementation of Byte Pair Encoding (BPE) Tokenizer."""
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.bpe_vocab = {}
        self.special_tokens = special_tokens or ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"]

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def build_vocab(self, corpus):
        words = self.preprocess(corpus)
        word_counts = Counter(words)

        self.bpe_vocab = {f"{token}": idx for idx, token in enumerate(self.special_tokens)}
        pairs = defaultdict(int)
        for word, freq in word_counts.items():
            tokens = list(word) + ["</w>"]
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += freq

        return word_counts, pairs

    def merge_pairs(self, pairs, word_counts):
        best_pair = max(pairs, key=pairs.get)
        merged_vocab = {}

        for word, freq in word_counts.items():
            word_str = " ".join(word)
            new_word = re.sub(f"{best_pair[0]} {best_pair[1]}", f"{best_pair[0]}{best_pair[1]}", word_str)
            merged_vocab[tuple(new_word.split())] = freq

        return merged_vocab

    def train(self, corpus):
        word_counts, pairs = self.build_vocab(corpus)
        for _ in range(self.vocab_size - len(self.bpe_vocab)):
            if not pairs:
                break
            word_counts = self.merge_pairs(pairs, word_counts)
            pairs = defaultdict(int)
            for word, freq in word_counts.items():
                tokens = word
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += freq

        for word in word_counts:
            self.bpe_vocab["".join(word)] = len(self.bpe_vocab)

    def encode(self, text):
        tokens = self.preprocess(text)
        encoded = []
        for token in tokens:
            if token in self.bpe_vocab:
                encoded.append(self.bpe_vocab[token])
            else:
                encoded.append(self.bpe_vocab["<|unk|>"])
        return encoded

    def decode(self, encoded):
        reverse_vocab = {idx: token for token, idx in self.bpe_vocab.items()}
        return " ".join(reverse_vocab[idx] for idx in encoded if idx in reverse_vocab)


@step
def load_data() -> dict:
    """Load and preprocess the dataset."""
    # Load the dataset from JSON
    with open("SW_EpisodeIV_VI.json", "r") as file:
        data = json.load(file)

    # Combine all dialogue lines into one corpus
    corpus = " ".join(entry["Line"] for entry in data)

    # Initialize and train the BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.train(corpus)

    # Encode all dialogue lines
    features = [tokenizer.encode(entry["Line"]) for entry in data]

    return {
        "features": features,
        "labels": features, 
        "tokenizer_vocab": tokenizer.bpe_vocab
    }


@step
def train_model(data: dict) -> None:
    """Mock training process using the tokenized data."""
    total_features = sum(len(seq) for seq in data["features"])
    total_labels = sum(len(seq) for seq in data["labels"])

    print(f"Trained model using {len(data['features'])} data points. "
          f"Total tokens in features: {total_features}, total tokens in labels: {total_labels}")
    print(f"Sample Vocabulary (first 10): {list(data['tokenizer_vocab'].items())[:10]}")


@pipeline
def tokenizer_pipeline():
    """Define a pipeline connecting data loading and training steps."""
    dataset = load_data()
    train_model(dataset)


if __name__ == "__main__":
    run = tokenizer_pipeline()

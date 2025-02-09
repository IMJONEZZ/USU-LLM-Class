# bpe_tokenizer.py
from collections import defaultdict
from transformers import AutoTokenizer

class BPE:
    """Byte-Pair Encoding: Subword-based tokenization algorithm."""

    def __init__(self, corpus, vocab_size):
        """Initialize BPE tokenizer."""
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}

    def train(self):
        """Train BPE tokenizer."""
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, _ in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        alphabet = sorted(set("".join(self.word_freqs.keys())))
        vocab = ["</w>"] + alphabet.copy()
        self.splits = {word: list(word) for word in self.word_freqs.keys()}

        while len(vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs()
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)

            merge_key = f"{best_pair[0]} {best_pair[1]}"
            self.merges[merge_key] = best_pair[0] + best_pair[1]
            self.splits = self.merge_pair(*best_pair)
            vocab.append(best_pair[0] + best_pair[1])
        return self.merges

    def compute_pair_freqs(self):
        """Compute the frequency of each pair of symbols in the corpus."""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b):
        """Merge the given pair."""
        for word in self.word_freqs:
            split = self.splits[word]
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def tokenize(self, text):
        """Tokenize a given text with trained BPE tokenizer."""
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, _ in pre_tokenize_result]
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            a, b = pair.split()
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits_text[idx] = split
        return sum(splits_text, [])

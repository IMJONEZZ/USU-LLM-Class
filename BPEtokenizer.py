import pandas as pd
import json
import re
from collections import Counter, defaultdict


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
            new_token_list = tuple(
                re.sub(pattern, "".join(pair), " ".join(token_list)).split()
            )
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
            pair_indices = [
                (self.str_to_int.get(p[0]), self.str_to_int.get(p[1]))
                for p in pairs
            ]
            best_pair = min(
                pair_indices,
                default=None,
                key=lambda x: x if None not in x else float("inf"),
            )
            if best_pair is None:
                break
            tokens = self.merge_tokens(tokens, best_pair)

        token_ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"])
            for token in tokens
        ]
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

import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.bpe_codes = {}
        self.bpe_codes_reverse = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def get_vocab(self, corpus):
        vocab = Counter()
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                tokenized_word = " ".join(list(word)) + " </w>"
                vocab[tokenized_word] += 1
        return vocab

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = p.sub("".join(pair), word)
            new_vocab[new_word] = freq
        return new_vocab

    def fit(self, corpus):
        vocab = self.get_vocab(corpus)
        for _ in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.bpe_codes[best] = len(self.bpe_codes)
            self.bpe_codes_reverse["".join(best)] = best

        # Create token-to-id and id-to-token mappings
        all_tokens = set()
        for word in vocab:
            all_tokens.update(word.split())
        all_tokens = sorted(all_tokens)  # Ensure consistent order
        self.token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text):
        words = re.findall(r"\w+(?:'\w+)?(?:')?|[^\w\s]", text, re.UNICODE)
        encoded_tokens = []
        for word in words:
            symbols = list(word) + ["</w>"]
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                pair_freqs = {pair: self.bpe_codes.get(pair, float("inf")) for pair in pairs}
                best_pair = min(pair_freqs, key=pair_freqs.get)
                if best_pair not in self.bpe_codes:
                    break
                symbols = [
                    symbols[i]
                    if i + 1 >= len(symbols) or (symbols[i], symbols[i + 1]) != best_pair
                    else "".join(best_pair)
                    for i in range(len(symbols))
                    if i == 0 or (symbols[i - 1], symbols[i]) != best_pair
                ]
            for symbol in symbols:
                encoded_tokens.append(self.token_to_id[symbol])
        return encoded_tokens

    def decode(self, token_ids):
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        text = "".join(tokens).replace("</w>", " ").strip()
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)  # Remove space before punctuation
        return text
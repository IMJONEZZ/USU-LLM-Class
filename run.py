import json
import re
from typing import Optional
from zenml import pipeline, step

# --------------------------------------------------
# THE SIMPLETOKENIZER CLASS (in the same file)
# --------------------------------------------------
class SimpleTokenizer:
    def __init__(
        self, 
        vocab: dict, 
        merges: Optional[list[tuple[str, str]]] = None
    ):
        """
        We provide a default empty list for merges so that the tokenizer
        will work even if merges are not explicitly passed in.
        """
        if merges is None:
            merges = []

        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

        # Convert list of merges to a dict for fast lookup: {("A", "B"): rank, ...}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    def _bpe_encode_word(self, word: str) -> list[str]:
        # Start as a list of individual characters
        tokens = list(word)

        # If the word is 1 char or empty, no merging is done
        if len(tokens) <= 1:
            return [word]

        # The standard BPE merge loop
        while True:
            # Create all adjacent pairs
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]

            # Find the highest-priority pair (lowest rank)
            min_rank = float('inf')
            min_pair = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            # If no merges can be done, break
            if min_pair is None:
                break

            # Otherwise, merge the highest priority pair in 'tokens'
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1 
                    and (tokens[i], tokens[i+1]) == min_pair
                ):
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2  # skip past the merged pair
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        # Split on punctuation, spaces, etc., keeping delimiters
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove empty strings and strip
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        final_tokens = []
        # Apply BPE to word-like tokens; keep punctuation as-is
        for item in preprocessed:
            # If purely punctuation, handle as one token:
            if re.match(r'^([,.:;?_!"()\']|--)$', item):
                final_tokens.append(item)
            else:
                # Apply BPE merges to the word token
                bpe_subwords = self._bpe_encode_word(item)
                final_tokens.extend(bpe_subwords)

        # Replace out-of-vocab tokens with <|unk|>
        final_tokens = [
            t if t in self.str_to_int else "<|unk|>"
            for t in final_tokens
        ]

        # Convert tokens to IDs
        ids = [self.str_to_int[token] for token in final_tokens]
        return ids

    def decode(self, ids: list[int]) -> str:
        # Convert IDs to tokens
        tokens = [self.int_to_str[i] for i in ids]
        # Join with space, then remove extra spacing before punctuation
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# --------------------------------------------------
# 1) LOAD DATA
# --------------------------------------------------
@step
def load_data() -> str:
    """
    Loads all 'Line' fields from the Star Wars JSON array and
    concatenates them into one big string.
    """
    with open("SW_EpisodeIV_VI.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [item["Line"] for item in data if "Line" in item]
    combined_text = " ".join(lines)
    return combined_text

# --------------------------------------------------
# 2) BUILD VOCABULARY
# --------------------------------------------------
@step
def build_vocab(text: str) -> dict:
    """
    Dynamically build a vocabulary dict {token: idx} from the given text.
    We split the text with the same pattern as SimpleTokenizer, then
    assign an integer ID to each unique token.
    """
    # Split text the same way as SimpleTokenizer
    split_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    split_tokens = [t.strip() for t in split_tokens if t.strip()]

    # Unique tokens (you could sort by frequency if needed)
    unique_tokens = sorted(set(split_tokens))

    # Create a vocab with <|unk|> as ID 0
    vocab = {"<|unk|>": 0}
    current_id = 1
    for token in unique_tokens:
        vocab[token] = current_id
        current_id += 1

    return vocab

# --------------------------------------------------
# 3) ENCODE THE TEXT
# --------------------------------------------------
@step
def encode_text(text: str, vocab: dict) -> list[int]:
    """
    Uses SimpleTokenizer to turn the text into a list of token IDs
    according to the dynamically built vocabulary.
    """
    # Note: merges not provided here, so we rely on the default empty merges
    tokenizer = SimpleTokenizer(vocab=vocab)
    token_ids = tokenizer.encode(text)
    return token_ids

# --------------------------------------------------
# 4) MOCK TRAIN
# --------------------------------------------------
@step
def train_model(token_ids: list[int]) -> None:
    """
    A mock 'training' step that just demonstrates how many tokens we received.
    Replace this with your actual training logic if desired.
    """
    print(f"Received {len(token_ids)} token IDs.")
    # Display the first 50 as a sanity check
    print("First 50 token IDs:", token_ids[:50])

# --------------------------------------------------
# PIPELINE DEFINITION
# --------------------------------------------------
@pipeline
def tokenizethetext():
    """
    Pipeline that loads text, builds vocab, tokenizes,
    and then mock-trains on the token IDs.
    """
    text = load_data()
    vocab = build_vocab(text)
    token_ids = encode_text(text, vocab)
    train_model(token_ids)

# --------------------------------------------------
# RUN THE PIPELINE
# --------------------------------------------------
if __name__ == "__main__":
    run = tokenizethetext()

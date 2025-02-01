import re
from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset

from dataloader import CustomDataLoader
from zenml import pipeline, step

# file has been reformatted to meet ruff expectations

class ImprovedTokenizer:
    #Tokenizer with preprocessing for lowercasing and contraction handling

    def __init__(self, vocab: dict, special_tokens: dict | None = None, max_length: int = 24):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.special_tokens = special_tokens or {}
        self._add_special_tokens(self.special_tokens)
        self.max_length = max_length  # Sets max sequence length

    def _add_special_tokens(self, tokens: dict) -> None:
        #Add special tokens to the vocabulary
        for token_symbol in tokens.values():
            new_id = len(self.str_to_int)  # Assign next available ID
            self.str_to_int[token_symbol] = new_id
            self.int_to_str[new_id] = token_symbol

    def add_special_tokens(self, tokens: dict) -> None:
        #Public method to add special tokens dynamically.
        self._add_special_tokens(tokens)

    def preprocess(self, text: str) -> str:
        #Lowercase text and expand common contractions
        text = text.lower()
        contraction_dict = {
            "m": "am", "s": "is", "t": "not", "re": "are",
            "ve": "have", "ll": "will", "d": "would",
            "N": "not", "n": "not", "c": "can"
        }

        def replace_contractions(match: re.Match) -> str:
            word, contraction = match.group(1), match.group(2)
            return f"{word} {contraction_dict.get(contraction, contraction)}"

        return re.sub(r"([a-zA-Z])'(m|s|t|re|ve|ll|d|N|n|c)(?=\s|$)", replace_contractions, text)

    def encode(self, text: str) -> list[int]:
        #Encode text into token IDs
        text = self.preprocess(text)
        tokens = [
            token.strip()
            for token in re.split(r"([,.:;?_!\"()']|--|\s)", text)
            if token.strip()
        ]
        tokens = [
            token if token in self.str_to_int else self.special_tokens.get("unk_token", "<|unk|>")
            for token in tokens
        ]
        token_ids = [self.str_to_int[token] for token in tokens]

        # Pad or truncate the sequence
        return token_ids[:self.max_length] + [self.str_to_int.get(self.special_tokens["pad_token"], 0)] * max(0, self.max_length - len(token_ids))

    def decode(self, ids: list[int]) -> str:
        #Decode token IDs back into text
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r"\s+([,.:;?\"()'])", r"\1", text)


@step
def load_and_tokenize_data() -> dict:
    #Load data, create vocabulary, and tokenize
    dataset = load_dataset(
        "andrewkroening/Star-wars-scripts-dialogue-IV-VI",
        split="train[:10%]",
    )

    vocab_list = [
        token.strip()
        for line in dataset["Line"]
        for token in re.split(r"([,.:;?_!\"()']|--|\s)", line)
        if token.strip()
    ]
    word_counts = Counter(vocab_list)
    vocab = {word: idx for idx, word in enumerate(word_counts.keys())}

    special_tokens = {
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "cls_token": "<|cls|>",
    }

    tokenizer = ImprovedTokenizer(vocab, special_tokens, max_length=50)

    #This is just for my own satisfaction to see the tokenizer at work."""
    first_line = dataset["Line"][0]
    print(f"Original: {first_line}")
    encoded = tokenizer.encode(first_line)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    inputs = [tokenizer.encode(line) for line in dataset["Line"]]
    max_length = 50
    padded_inputs = [seq[:max_length] + [0] * (max_length - len(seq)) for seq in inputs]

    unique_labels = list(set(dataset["Character"]))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_int[label] for label in dataset["Character"]]

    return {"inputs": padded_inputs, "labels": labels}


@step
def train_model(data: dict) -> None:
    #Train model using the dataset
    inputs = torch.tensor(data["inputs"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    dataset = TensorDataset(inputs, labels)
    dataloader = CustomDataLoader(dataset, batch_size=10, shuffle=True)

    for batch_idx, (inputs_batch, labels_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Inputs shape: {inputs_batch.shape}, Labels shape: {labels_batch.shape}")


@pipeline
def my_pipeline_with_tokenization() -> None:
    #Define a pipeline that connects the steps.
    data = load_and_tokenize_data()
    train_model(data)


if __name__ == "__main__":
    my_pipeline_with_tokenization()

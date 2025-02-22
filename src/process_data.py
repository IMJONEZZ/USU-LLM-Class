import re
from collections import Counter

from datasets import load_dataset
from datasets import DatasetDict

from tokenizer import ImprovedTokenizer


def process_data():
    dataset = load_dataset(
        "andrewkroening/Star-wars-scripts-dialogue-IV-VI",
        split="train[:10%]",
    )

    # Create label mapping using the full dataset
    unique_labels = list(set(dataset["Character"]))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Split the dataset into training, validation, and testing sets
    dataset = dataset.train_test_split(
        test_size=0.2, seed=42
    )  # 80% training, 20% testing
    train_test_valid = dataset["train"].train_test_split(
        test_size=0.25, seed=42
    )  # 60% training, 20% validation, 20% testing

    dataset = DatasetDict(
        {
            "train": train_test_valid["train"],
            "validation": train_test_valid["test"],
            "test": dataset["test"],
        }
    )

    # Create vocabulary and tokenizer
    vocab_list = [
        token.strip()
        for line in dataset["train"]["Line"]
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

    tokenizer = ImprovedTokenizer(vocab, special_tokens, max_length=24)
    vocab_size = len(tokenizer.str_to_int)

    # Tokenize the datasets
    def tokenize_dataset(dataset):
        inputs = [tokenizer.encode(line) for line in dataset["Line"]]
        max_length = 24
        padded_inputs = [
            seq[:max_length] + [0] * (max_length - len(seq)) for seq in inputs
        ]
        labels = [label_to_int[label] for label in dataset["Character"]]
        return {"inputs": padded_inputs, "labels": labels}

    train_data = tokenize_dataset(dataset["train"])
    validation_data = tokenize_dataset(dataset["validation"])
    test_data = tokenize_dataset(dataset["test"])

    processed_data = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
        "vocab_size": vocab_size,
        "label_to_int": label_to_int,  # Label mapping
        "vocab": vocab,  # ✅ Store vocab for tokenizer initialization
        "special_tokens": special_tokens,  # ✅ Store special tokens
    }

    return processed_data

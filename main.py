from zenml import pipeline, step
import json
import re
import collections
import torch
from torch.utils.data import Dataset, DataLoader


@step
def load_data(file_path: str) -> list:
    """Loads JSON data from a file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


@step
def build_vocab(data: list) -> dict:
    """Creates a vocabulary from preprocessed text."""
    preprocessed = []
    for entry in data:
        split_text = re.split(r'([,.:;?_!"()\']|--|\s)', entry["Line"])
        split_text = [item.strip() for item in split_text if item.strip()]
        preprocessed.extend(split_text)

    # Build Vocabulary
    all_tokens = sorted(set(preprocessed))
    pairs = []
    for word in all_tokens:
        word = list(word)
        if len(word) > 1:
            count = 0
            while count < len(word) - 1:
                pairs.append(word[count] + word[count + 1])
                count += 1

    pairs.append("<|unk|>")
    vocab = {token: idx for idx, token in enumerate(sorted(set(pairs)))}
    return vocab


@step
def tokenize_text(data: list, vocab: dict) -> list:
    """Tokenizes text using a simple tokenizer."""

    class SimpleTokenizer:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            pairs = []
            for word in preprocessed:
                word = list(word)
                if len(word) > 1:
                    count = 0
                    while count < len(word) - 1:
                        pairs.append(word[count] + word[count + 1])
                        count += 1
            pairs1 = [item if item in self.str_to_int else "<|unk|>" for item in pairs]
            ids = [self.str_to_int[s] for s in pairs1]
            return ids

    tokenizer = SimpleTokenizer(vocab)
    tokenized_text = []
    for entry in data:
        tokenized_text.extend(tokenizer.encode(entry["Line"]))
    return tokenized_text


class TextDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


@step
def create_dataloader_v1(
    token_ids: list, batch_size: int = 4, max_length: int = 256, stride: int = 128
) -> DataLoader:
    """Creates a DataLoader from tokenized text."""
    dataset = TextDataset(token_ids, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    return dataloader


@pipeline
def assignment_3_pipeline(file_path: str):
    """ZenML pipeline for text processing."""
    data = load_data(file_path)
    vocab = build_vocab(data)
    tokenized_text = tokenize_text(data, vocab)
    dataloader = create_dataloader_v1(tokenized_text)

    return dataloader


if __name__ == "__main__":
    assignment_3_pipeline(file_path="SW_EpisodeIV_VI.json")

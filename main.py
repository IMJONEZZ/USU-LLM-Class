from zenml import pipeline, step
import json
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


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
def tokenize_text(data: list):
    """Tokenizes text using BERT's tokenizer and returns tokenized tensors."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    texts = [entry["Line"] for entry in data]
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return encoding


class TextDataset(Dataset):
    def __init__(self, encoding):
        self.input_ids = encoding["input_ids"]
        self.attention_mask = encoding["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


@step
def create_dataloader_v1(encoding: dict, batch_size: int = 4) -> DataLoader:
    """Creates a DataLoader from BERT-tokenized text."""
    dataset = TextDataset(encoding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


@pipeline
def assignment_4_pipeline(file_path: str):
    """ZenML pipeline for text processing using BERT."""
    data = load_data(file_path)
    encoding = tokenize_text(data)
    dataloader = create_dataloader_v1(encoding)
    return dataloader


if __name__ == "__main__":
    assignment_4_pipeline(file_path="SW_EpisodeIV_VI.json")

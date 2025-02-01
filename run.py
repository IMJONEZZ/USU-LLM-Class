import json
from typing import Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from zenml import pipeline, step


# --------------------------------------------------
# THE SIMPLETOKENIZER CLASS
# --------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab: dict, merges: Optional[list[tuple[str, str]]] = None):
        if merges is None:
            merges = []
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int.get(token, 0) for token in text.split()]  # 0 is <UNK>

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.int_to_str.get(i, "<UNK>") for i in ids])


# --------------------------------------------------
# 1) LOAD DATA
# --------------------------------------------------
@step
def load_data() -> str:
    with open("SW_EpisodeIV_VI.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [item["Line"] for item in data if "Line" in item]
    return " ".join(lines)


# --------------------------------------------------
# 2) BUILD VOCABULARY
# --------------------------------------------------
@step
def build_vocab(text: str) -> dict:
    tokens = text.split()
    vocab = {"<UNK>": 0}  # Add <UNK> token
    vocab.update({token: i for i, token in enumerate(sorted(set(tokens)), start=1)})
    return vocab


# --------------------------------------------------
# 3) ENCODE TEXT
# --------------------------------------------------
@step
def encode_text(text: str, vocab: dict) -> List[List[int]]:
    sentences = text.split(".")
    tokenizer = SimpleTokenizer(vocab=vocab)
    return [tokenizer.encode(sentence) for sentence in sentences if sentence.strip()]


# --------------------------------------------------
# 4) CUSTOM DATASET CLASS
# --------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self, tokenized_sequences: List[List[int]]):
        self.tokenized_sequences = tokenized_sequences

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        return torch.tensor(
            self.tokenized_sequences[idx], dtype=torch.long
        )  # No padding here


# --------------------------------------------------
# 5) COLLATE FUNCTION FOR PADDING
# --------------------------------------------------
def collate_fn(batch, max_len=50):
    """
    Pads batch sequences to a fixed max length (50).
    """
    batch = [torch.as_tensor(seq, dtype=torch.long) for seq in batch]

    # Pad all sequences to max_len manually
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)

    # Ensure every sequence is exactly max_len
    if padded_batch.shape[1] < max_len:
        pad_amount = max_len - padded_batch.shape[1]
        padded_batch = torch.cat(
            [
                padded_batch,
                torch.zeros((padded_batch.shape[0], pad_amount), dtype=torch.long),
            ],
            dim=1,
        )
    elif padded_batch.shape[1] > max_len:
        padded_batch = padded_batch[:, :max_len]  # Truncate if needed

    return padded_batch


# --------------------------------------------------
# 6) TRAIN MODEL WITH IMPROVED DATALOADER
# --------------------------------------------------
@step
def train_model(
    token_ids: List[List[int]], batch_size: int = 8, num_workers: int = 2
) -> None:
    dataset = TokenDataset(token_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Total sequences: {len(dataset)}")
    print(f"Total batches: {len(dataloader)}")

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: {batch.shape}")  # Dynamic shape


# --------------------------------------------------
# PIPELINE DEFINITION
# --------------------------------------------------
@pipeline
def tokenizethetext():
    text = load_data()
    vocab = build_vocab(text)
    token_ids = encode_text(text, vocab)
    train_model(token_ids)


# --------------------------------------------------
# RUN THE PIPELINE
# --------------------------------------------------
if __name__ == "__main__":
    tokenizethetext()

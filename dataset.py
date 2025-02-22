import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from zenml import step


class NextTokenDataset(Dataset):
    """Dataset for next-token prediction."""

    def __init__(self, tokenized_sequences: List[List[int]]):
        self.x_data = []
        self.y_data = []
        for seq in tokenized_sequences:
            x = seq[:-1]
            y = seq[1:]
            self.x_data.append(torch.tensor(x, dtype=torch.long))
            self.y_data.append(torch.tensor(y, dtype=torch.long))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def collate_fn(batch, max_len=50):
    """Pads sequences to max length."""
    xs, ys = zip(*batch)
    padded_x = pad_sequence(xs, batch_first=True, padding_value=0)
    padded_y = pad_sequence(ys, batch_first=True, padding_value=-100)

    if padded_x.shape[1] < max_len:
        pad_amount = max_len - padded_x.shape[1]
        padded_x = torch.cat(
            [padded_x, torch.zeros((padded_x.shape[0], pad_amount), dtype=torch.long)],
            dim=1,
        )

    if padded_y.shape[1] < max_len:
        pad_amount = max_len - padded_y.shape[1]
        padded_y = torch.cat(
            [
                padded_y,
                torch.full((padded_y.shape[0], pad_amount), -100, dtype=torch.long),
            ],
            dim=1,
        )

    return padded_x, padded_y


@step
def split_data(
    tokenized_sequences: List[List[int]], train_ratio=0.7, val_ratio=0.15
) -> Dict[str, List[List[int]]]:
    """Splits data into train, validation, and test sets."""
    total = len(tokenized_sequences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": tokenized_sequences[:train_end],
        "val": tokenized_sequences[train_end:val_end],
        "test": tokenized_sequences[val_end:],
    }

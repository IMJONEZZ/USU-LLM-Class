import torch
from torch.utils.data import Dataset

#Dataset
class StarWarsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = [self.tokenizer.encode(line) for line in data["Line"]]
        self.labels = data["Character"]

        # Pad sequences
        self.inputs = [seq[:max_length] + [0] * (max_length - len(seq)) for seq in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), self.labels[idx]

#Dataloader

from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=10, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

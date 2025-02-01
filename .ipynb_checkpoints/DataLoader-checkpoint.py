import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list or array-like): The dataset to be loaded.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            Tensor: A single data sample at index `idx`
        """
        return torch.tensor(self.data[idx], dtype=torch.float32)


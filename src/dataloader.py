import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

#Dataset
# class StarWarsDataset(Dataset):
#     def __init__(self, data, inputs, labels, max_length):
#
#         self.max_length = max_length
#         self.inputs = inputs
#         self.labels = labels
#
#         # Pad sequences
#         self.inputs = [seq[:max_length] + [0] * (max_length - len(seq)) for seq in self.inputs]
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         return torch.tensor(self.inputs[idx]), self.labels[idx]

#Dataloader

# def create_dataloader(dataset, batch_size=10, shuffle=True):
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class CustomDataLoader:
    """Custom DataLoader to mimic PyTorch's functionality with more flexibility."""

    def __init__(self, dataset, batch_size=10, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        #allows random shuffling of dataset before iterating
        self.shuffle = shuffle
        #creates a indices list of index positions of items in dataset
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        """Creates an iterator over the dataset."""
        if self.shuffle:
            torch.manual_seed(42)  
            self.indices = torch.randperm(len(self.dataset)).tolist()

        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start:start + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def collate_fn(self, batch):
        """Collates a batch of data into tensors (similar to PyTorch's default behavior)."""
        inputs, labels = zip(*batch)
        return torch.stack(inputs), torch.tensor(labels)

    def __len__(self):
        """Returns number of batches per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

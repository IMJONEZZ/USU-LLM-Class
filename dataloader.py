from zenml import step
from torch.utils.data import Dataset, DataLoader


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
def create_dataloader(encoding: dict, batch_size: int = 4) -> DataLoader:
    """Creates a DataLoader from BERT-tokenized text."""
    dataset = TextDataset(encoding)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

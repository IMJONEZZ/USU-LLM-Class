from zenml import pipeline, step
import json
import torch
from torch.utils.data import Dataset, DataLoader


class SWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@step
def load_data() -> dict:
    """Loads the dataset from a JSON file and prepares a DataLoader."""
    with open("SW_EpisodeIV_VI.json", "r") as f:
        data = json.load(f)

    dataset = SWDataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    return {"dataloader": dataloader}


@pipeline
def data_process_pipeline():
    """Define a pipeline that connects the steps."""
    load_data()


if __name__ == "__main__":
    run = data_process_pipeline()

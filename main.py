import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from zenml import pipeline, step


class StarWarsDataset(Dataset):
    def __init__(self, file):
        with open(file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        return item


@step
def dataloader(path) -> DataLoader:
    dataset = StarWarsDataset(path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


@pipeline
def pipeline():
    dataloader("SW_EpisodeIV_VI.json")


if __name__ == "__main__":
    run = pipeline()

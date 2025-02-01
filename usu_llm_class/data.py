import json
from torch.utils.data import Dataset


class StarWarsDataset(Dataset):
    def __init__(self, json_file):
        # Load the data from the JSON file
        with open(json_file, "r") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample at index idx.
        sample = self.data[idx]

        return sample

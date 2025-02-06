import json
from torch.utils.data import Dataset

class StarWarsDataset(Dataset):
    def __init__(self, json_file):
        # Load the data from the JSON file
        with open(json_file, "r") as file:
            self.dat = json.load(file)

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, i):
        # Get the sample at index i.
        sample = self.dat[i]

        return sample
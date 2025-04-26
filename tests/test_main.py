from main import StarWarsDataset, dataloader
from torch.utils.data import DataLoader


def test_star_wars_dataset():
    dataset = StarWarsDataset("SW_EpisodeIV_VI.json")
    assert len(dataset) == 84  # Verify the dataset length matches the JSON file
    assert dataset[0]["Line"] == "Did you hear that?  They've shut down the main reactor.  We'll be destroyed for sure.  This is madness!"
    assert dataset[83]["Line"] == "Quiet down will ya!  You got a mouth bigger than a meteor crater!"

def test_dataloader():
    loader = dataloader("SW_EpisodeIV_VI.json")
    assert isinstance(loader, DataLoader)  # Ensure it returns a DataLoader object
    batch = next(iter(loader))
    assert len(batch) <= 32  # Verify batch size is correct
    assert "Line" in batch[0]  # Ensure batch items have the expected structure
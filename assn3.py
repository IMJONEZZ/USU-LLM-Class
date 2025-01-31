from zenml import pipeline, step
from torch.utils.data import DataLoader
import torch

from tokenizer import make_tokenizer, get_string


# # Define the TextDataset and create_dataloader_v1 as before
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"}) # This is how you should do it
        token_ids = tokenizer.encode(txt)  # This is how you should do it

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# class TextDataset(torch.utils.data.Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):
#         self.input_ids = []
#         self.target_ids = []

#         token_ids = tokenizer.encode(txt)

#         # Create input and target sequences
#         for i in range(0, len(token_ids) - max_length, stride):
#             input_chunk = token_ids[i : i + max_length]
#             target_chunk = token_ids[i + 1 : i + max_length + 1]
#             self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
#             self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         # Ensure proper slicing and consistent data return format
#         return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = make_tokenizer(
        txt
    )  # Replace None with your actual tokenizer initialization

    dataset = TextDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


@step
def load_data(file_path) -> DataLoader:
    """Simulates loading training data and labels from a file and returns a DataLoader."""

    # Load text data from the file (assuming 'number-data.txt' is available)
    txt = get_string(file_path)

    # Create a DataLoader from the text data
    dataloader = create_dataloader_v1(txt)

    return dataloader


@step
def train_model(dataloader: DataLoader) -> None:
    """Mock 'training' process that demonstrates using the input data."""

    for batch_idx, (input_chunk, target_chunk) in enumerate(dataloader):
        # Just an example; typically you'd perform actual model training here
        if batch_idx == 0:
            print(f"Training with batch {batch_idx + 1}:")
            print(f"Input: {input_chunk}")
            print(f"Target: {target_chunk}")


@pipeline
def assn3_pipe():
    """Define a pipeline that connects the steps."""
    file_path = "SW_EpisodeIV_VI.json"  # Example path to your input data

    #### dataloader =
    load_data(file_path)

    # train_model(dataloader)


if __name__ == "__main__":
    run = assn3_pipe()
    # You can now use the `run` object to see steps, outputs, etc.

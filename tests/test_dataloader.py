import pytest
import torch
from torch.utils.data import DataLoader

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# os.path.join(__file__, "mock_file.txt")

# from tokenizer import make_tokenizer, get_string
from assn3 import TextDataset, create_dataloader_v1


# Mock tokenizer for testing
class MockTokenizer:
    def encode(self, txt):
        # Simulates encoding a string into token IDs (e.g., 0-9 repeated)
        return list(range(len(txt)))


# Mock the tokenizer creation function
def make_tokenizer(txt=None):
    return MockTokenizer()


# Test the TextDataset class
def test_text_dataset():
    txt = """Lorem ipsum dolor sit amet."""
    tokenizer = make_tokenizer(txt)
    max_length = 5
    stride = 2

    dataset = TextDataset(txt, tokenizer, max_length, stride)

    # Check the number of samples
    expected_length = (len(tokenizer.encode(txt)) - max_length) // stride
    assert len(dataset) == expected_length, "Incorrect dataset length."

    # Check individual items
    for idx, (input_chunk, target_chunk) in enumerate(dataset):
        assert len(input_chunk) == max_length, "Input chunk length mismatch."
        assert len(target_chunk) == max_length, "Target chunk length mismatch."
        assert torch.is_tensor(input_chunk), "Input is not a tensor."
        assert torch.is_tensor(target_chunk), "Target is not a tensor."


# Test the create_dataloader_v1 function
def test_create_dataloader():
    txt = """Lorem ipsum dolor sit amet."""
    batch_size = 4
    max_length = 5
    stride = 2
    dataloader = create_dataloader_v1(
        txt, batch_size=batch_size, max_length=max_length, stride=stride
    )

    # Verify the DataLoader batches
    for batch_idx, (input_chunks, target_chunks) in enumerate(dataloader):
        assert input_chunks.shape[0] == batch_size, "Batch size mismatch."
        assert target_chunks.shape[0] == batch_size, "Batch size mismatch."
        assert input_chunks.shape[1] == max_length, "Input chunk length mismatch."
        assert target_chunks.shape[1] == max_length, "Target chunk length mismatch."


# Mock get_string for testing
def get_string(file_path):
    return "Lorem ipsum dolor sit amet."


# # Test the load_data step
def test_load_data():
    # file_path = "mock_file.txt"
    # dataloader = load_data(file_path)

    with open("mock_file.txt", "r") as file:
        text = file.read()

    dataloader = create_dataloader_v1(text)

    # Verify that the returned object is a DataLoader
    assert isinstance(dataloader, DataLoader), "load_data did not return a DataLoader."

    # Verify DataLoader contents
    for batch_idx, (input_chunks, target_chunks) in enumerate(dataloader):
        assert input_chunks.shape[1] == 256, "Input chunk length mismatch."
        assert target_chunks.shape[1] == 256, "Target chunk length mismatch."
        break  # Check only the first batch for simplicity


# def test_short_text():
#     txt = "Hi."
#     batch_size = 1
#     max_length = 5
#     stride = 2
#     dataloader = create_dataloader_v1(txt, batch_size=batch_size, max_length=max_length, stride=stride)

#     # Verify that the DataLoader handles short text
#     for batch_idx, (input_chunks, target_chunks) in enumerate(dataloader):
#         assert input_chunks.shape[0] == 1, "Batch size mismatch for short text."
#         assert target_chunks.shape[0] == 1, "Batch size mismatch for short text."
#         break


def test_batch_size_1():
    txt = "Lorem ipsum dolor sit amet."
    batch_size = 1
    max_length = 5
    stride = 2
    dataloader = create_dataloader_v1(
        txt, batch_size=batch_size, max_length=max_length, stride=stride
    )

    # Verify that the DataLoader handles batch size 1 correctly
    for batch_idx, (input_chunks, target_chunks) in enumerate(dataloader):
        assert input_chunks.shape[0] == 1, "Batch size mismatch for batch size 1."
        assert target_chunks.shape[0] == 1, "Batch size mismatch for batch size 1."
        break


def test_multiple_epochs():
    with open("mock_file.txt", "r") as file:
        txt = file.read()

    batch_size = 2
    max_length = 5
    stride = 2
    dataloader = create_dataloader_v1(
        txt, batch_size=batch_size, max_length=max_length, stride=stride
    )

    epoch_count = 0
    for batch_idx, (input_chunks, target_chunks) in enumerate(dataloader):
        epoch_count += 1
        if epoch_count > 2:
            break  # Only one batch per epoch for simplicity

    # Check if multiple epochs are handled properly
    assert epoch_count >= 2, "DataLoader did not iterate over multiple epochs."


if __name__ == "__main__":
    pytest.main()

# tests/test_evalutor.py

import pytest
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from run import (
    SimpleLanguageModel,
    NextTokenDataset,
    collate_fn,
    evaluate,
    gather_detailed_eval_info,
)


@pytest.fixture
def dummy_data():
    """
    Returns a small list of tokenized sequences for testing.
    Each sequence is a list of token IDs, e.g. [5, 10, 15].
    """
    return [
        [1, 2, 3],
        [4, 5, 6, 7],
        [10, 11],
    ]


@pytest.fixture
def dummy_dataset(dummy_data):
    """
    Creates a NextTokenDataset using the dummy_data fixture.
    """
    return NextTokenDataset(dummy_data)


@pytest.fixture
def dummy_dataloader(dummy_dataset):
    """
    Creates a DataLoader (batch_size=2) for testing evaluate() and gather_detailed_eval_info().
    """
    return DataLoader(dummy_dataset, batch_size=2, collate_fn=collate_fn)


@pytest.fixture
def dummy_model():
    """
    Creates a SimpleLanguageModel with a small vocab_size for testing.
    """
    vocab_size = 20  # Larger than max token in dummy_data
    embed_dim = 8
    return SimpleLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim)


def test_next_token_dataset_length(dummy_dataset, dummy_data):
    """
    Checks if NextTokenDataset length matches the number of sequences passed in.
    """
    assert len(dummy_dataset) == len(dummy_data), (
        "Dataset length should match the number of sequences."
    )


def test_next_token_dataset_contents(dummy_dataset):
    """
    Verifies that each x/y pair is formed correctly:
      If sequence is [1, 2, 3], then x=[1, 2], y=[2, 3].
    """
    for idx, (x_tensor, y_tensor) in enumerate(dummy_dataset):
        original_seq = dummy_dataset.x_data[idx].tolist() + [dummy_dataset.y_data[idx][-1].item()]  # reconstruct
        # Example: x=[1,2], y=[2,3], so combined = [1,2,3]

        # Check that x and y line up properly
        # x[i] should match original_seq[i], y[i] == original_seq[i+1]
        for i, x_val in enumerate(x_tensor.tolist()):
            assert x_val == original_seq[i], f"x mismatch at sample {idx}, position {i}"
        for i, y_val in enumerate(y_tensor.tolist()):
            assert y_val == original_seq[i + 1], f"y mismatch at sample {idx}, position {i}"


def test_collate_fn_shape(dummy_dataset):
    """
    Checks that collate_fn produces padded tensors of the correct shape.
    """
    dataloader = DataLoader(dummy_dataset, batch_size=2, collate_fn=collate_fn)
    for x_batch, y_batch in dataloader:
        # Defaults to pad up to seq_len=50 in run.py
        assert x_batch.shape[1] == 50, "X should be padded to length 50"
        assert y_batch.shape[1] == 50, "Y should be padded to length 50"
        break  # Only need to check the first batch


def test_evaluate_returns_float(dummy_dataloader, dummy_model):
    """
    Evaluate() should return a float loss value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dummy_model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loss = evaluate(model, dummy_dataloader, criterion, device)
    assert isinstance(loss, float), "evaluate() must return a float loss value."


def test_evaluate_no_nan_infs(dummy_dataloader, dummy_model):
    """
    Check that evaluate() doesn't produce NaN or Inf.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dummy_model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loss = evaluate(model, dummy_dataloader, criterion, device)
    assert not torch.isnan(torch.tensor(loss)), "Loss should not be NaN."
    assert not torch.isinf(torch.tensor(loss)), "Loss should not be Inf."


def test_gather_detailed_eval_info(dummy_dataloader, dummy_model):
    """
    Ensures gather_detailed_eval_info returns detailed info about each token,
    including whether it was predicted correctly and the confidence in the true token.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dummy_model.to(device)

    # Forward pass once so the model isn't uninitialized
    # (this is optional, but ensures embeddings are properly set).
    # Normally, an untrained model is fine for the test, though.
    for x_batch, y_batch in dummy_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _ = model(x_batch)
        break

    # Call gather_detailed_eval_info
    results = gather_detailed_eval_info(model, dummy_dataloader, device)

    # 'results' is a list of sequences, each a list of dict entries (one per token)
    assert isinstance(results, list), "Should return a list of sequences."

    # We expect at least as many sequences as dummy_data, though they may be batched
    # so let's just check the structure of the first sequence returned.
    assert len(results) > 0, "Should have at least one sequence of results."

    # Each item in 'results' should be a list describing each token in that sequence
    first_sequence = results[0]
    assert isinstance(first_sequence, list), "Each sequence entry should be a list."
    if len(first_sequence) > 0:
        # Check the structure of a single token info
        token_info = first_sequence[0]
        expected_keys = {
            "true_token_id",
            "pred_token_id",
            "confidence_in_true_token",
            "is_correct",
        }
        assert set(token_info.keys()) == expected_keys, "Missing or extra keys in token info."

        # Confidence should be between 0 and 1
        conf = token_info["confidence_in_true_token"]
        assert 0.0 <= conf <= 1.0, "Confidence should be between 0 and 1."

        # is_correct should be a bool
        assert isinstance(token_info["is_correct"], bool), "is_correct should be boolean."

    # This verifies the structure is as intended for potential 'smart learning' usage.

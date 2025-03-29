import pytest
import torch
import numpy as np
import faiss
from vectordb import get_embeddings


@pytest.fixture
def mock_encoding():
    """Creates a mock encoding dictionary with tokenized tensors."""
    return {
        "input_ids": torch.randint(0, 100, (5, 10)),  # 5 sequences of length 10
        "attention_mask": torch.ones((5, 10)),  # Full attention
    }


@pytest.fixture
def mock_query_encoding():
    """Creates a mock query encoding dictionary."""
    return {
        "input_ids": torch.randint(0, 100, (1, 10)),  # 1 query sequence of length 10
        "attention_mask": torch.ones((1, 10)),  # Full attention
    }


def test_get_embeddings(mock_encoding):
    """Test that get_embeddings produces valid embeddings."""
    embeddings = get_embeddings(mock_encoding)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 5  # Should match batch size
    assert embeddings.ndim == 2  # Should be (batch_size, embedding_dim)


def test_vectordb(mock_encoding, mock_query_encoding):
    """Test that vectordb can successfully perform a similarity search."""
    dataset_embeddings = get_embeddings(mock_encoding)
    query_embeddings = get_embeddings(mock_query_encoding)

    assert (
        dataset_embeddings.shape[1] == query_embeddings.shape[1]
    )  # Ensure dimensions match

    # Check that FAISS indexing and searching do not crash
    d = dataset_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(dataset_embeddings)
    k = 3
    distances, indices = index.search(query_embeddings, k)

    assert distances.shape == (1, k)  # Expect one query result with k neighbors
    assert indices.shape == (1, k)  # Expect indices for k nearest neighbors

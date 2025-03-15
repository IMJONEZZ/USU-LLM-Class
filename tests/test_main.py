
import pytest
from sentence_transformers import SentenceTransformer
from unittest.mock import MagicMock
from datasets import Dataset


# Mock Pinecone client and index
@pytest.fixture(scope="module")
def mock_pinecone():
    """Fixture to mock the Pinecone client and index."""

    # Mock the Pinecone Index object
    mock_index = MagicMock()

    # Mock describe_index_stats to return empty data initially
    mock_index.describe_index_stats.return_value = {"total_vector_count": 0}

    # Mock upsert method
    mock_index.upsert.return_value = None

    # Mock query method to return fake results
    mock_index.query.return_value = {
        "matches": [
            {"id": "1", "score": 0.98, "metadata": {"text": "Example match 1"}},
            {"id": "2", "score": 0.96, "metadata": {"text": "Example match 2"}},
        ]
    }

    yield mock_index


# Mock dataset
@pytest.fixture
def mock_dataset():
    """Fixture to mock the dataset."""
    data = {
        "id": [1, 2, 3],
        "text": [
            "What is the capital of France?",
            "Who is the president of the United States?",
            "What is the boiling point of water?",
        ],
    }
    return Dataset.from_dict(data)


def test_index_creation(mock_pinecone):
    """Test if the index is created successfully."""
    index = mock_pinecone
    stats = index.describe_index_stats()
    assert stats["total_vector_count"] == 0  # Ensure no vectors are added yet


def test_upsert_data(mock_pinecone, mock_dataset):
    """Test if data can be upserted into the Pinecone index."""
    index = mock_pinecone

    # Convert dataset into batches manually for upserting
    batch_size = 1  # Change the batch size to suit your needs
    for i in range(0, len(mock_dataset), batch_size):
        batch = mock_dataset[i : i + batch_size]  # Slice the dataset into batches
        index.upsert(batch)

    # Since this is a mock test, the total vector count should still be 0
    stats = index.describe_index_stats()
    assert (
        stats["total_vector_count"] == 0
    )  # Mock data isn't actually inserted in this test


def test_query_data(mock_pinecone):
    """Test if querying works with a sample text."""
    index = mock_pinecone
    model_name = "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
    model = SentenceTransformer(model_name)

    # Example query
    query_text = "Who is the current US president?"
    query_embedding = model.encode(query_text).tolist()

    # Perform the query using the mock index
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Validate that query returns matches
    assert len(results["matches"]) > 0  # Ensure we get some matches
    assert results["matches"][0]["score"] > 0.9  # Example to check match score
    assert "metadata" in results["matches"][0]  # Ensure metadata is present

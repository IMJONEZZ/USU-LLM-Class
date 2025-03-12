import pytest
from unittest.mock import MagicMock
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings

# Import the functions we need from rileyassn7
from rileyassn7 import query_vector_db


@pytest.fixture
def mock_embedding_model():
    """Fixture to mock the embedding model."""
    mock_model = MagicMock()
    mock_model.embed_query.side_effect = lambda text: [
        float(ord(c)) for c in text
    ]  # Mock embeddings as a list of ASCII values
    return mock_model


@pytest.fixture
def mock_vector_store(mock_embedding_model):
    """Fixture to mock a vector database."""
    mock_store = MagicMock(spec=Chroma)

    # Simulated response for similarity search
    mock_store.similarity_search_by_vector.side_effect = lambda embedding, k: [
        MagicMock(page_content="Mock response 1"),
        MagicMock(page_content="Mock response 2"),
        MagicMock(page_content="Mock response 3"),
    ][:k]

    return mock_store


def test_embedding_generation(mock_embedding_model):
    """Test if the embedding function correctly converts text to a vector."""
    text = "Jedi"
    embedding = mock_embedding_model.embed_query(text)

    assert isinstance(embedding, list)
    assert all(
        isinstance(val, float) for val in embedding
    )  # Ensure all elements are numbers


def test_vector_db_query(mock_vector_store, mock_embedding_model):
    """Test querying the vector database using mocks."""
    query_text = "Where is the Force?"

    # Use the mocked function
    results = query_vector_db(query_text, k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0] == "Mock response 1"  # Ensure mocked data is returned
    assert results[1] == "Mock response 2"


def test_empty_query(mock_vector_store, mock_embedding_model):
    """Ensure an empty query does not break the function."""
    results = query_vector_db("", k=1)

    assert isinstance(results, list)  # Should return an empty list or mock response


if __name__ == "__main__":
    pytest.main()

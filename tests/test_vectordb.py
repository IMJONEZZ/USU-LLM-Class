# import sys
# import os
import pytest
from langchain_community.vectorstores import Chroma
from rileyassn7 import embed_model, query_vector_db


# from langchain_community.embeddings import HuggingFaceEmbeddings
# import pandas as pd

# Import functions from your script


@pytest.fixture(scope="module")
def sample_data():
    """Fixture to set up sample Yoda texts and embeddings."""
    return [
        "Train yourself to let go of everything you fear to lose.",
        "Difficult to see. Always in motion is the future.",
        "Wars not make one great.",
    ]


@pytest.fixture(scope="module")
def setup_vector_db(sample_data):
    """Fixture to initialize a fresh vector database with sample data."""
    # Create a new vector store for testing
    test_vector_store = Chroma(
        persist_directory="./test_chroma_db", embedding_function=embed_model
    )

    # Add sample data
    for i, text in enumerate(sample_data):
        test_vector_store.add_texts(
            texts=[text],
            metadatas=[{"character": "YODA"}],
            ids=[str(i)],
            embeddings=[embed_model.embed_query(text)],
        )

    test_vector_store.persist()
    return test_vector_store


def test_vector_db_initialization(setup_vector_db):
    """Test if the vector database initializes correctly."""
    assert setup_vector_db is not None


def test_embedding_generation(sample_data):
    """Test if embeddings are being generated correctly."""
    embeddings = [embed_model.embed_query(text) for text in sample_data]
    assert all(isinstance(emb, list) and len(emb) > 0 for emb in embeddings)


def test_data_insertion(setup_vector_db, sample_data):
    """Check if embeddings and texts were successfully inserted."""
    results = setup_vector_db.similarity_search(sample_data[0], k=1)
    assert len(results) > 0
    assert results[0].page_content == sample_data[0]  # Should match the inserted text


def test_query_vector_db(setup_vector_db):
    """Test querying functionality."""
    query_text = "Future is uncertain."
    results = query_vector_db(query_text, k=1)
    assert len(results) > 0  # Ensure it returns results
    assert isinstance(results[0], str)  # Should return text


def test_empty_query():
    """Ensure empty query does not break the function."""
    results = query_vector_db("", k=1)
    assert isinstance(results, list)  # Should return an empty list or valid response


if __name__ == "__main__":
    pytest.main()

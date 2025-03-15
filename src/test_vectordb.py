#attempted fix at pytests

import os
import pytest
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load API key securely
API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")

pc = Pinecone(api_key=API_KEY)
INDEX_NAME = "test-vector-db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def pinecone_index():
    """Setup Pinecone index for testing."""
    if INDEX_NAME not in pc.list_indexes():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, 
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )

    index = pc.Index(INDEX_NAME)
    yield index  # Provide index to tests

    # Cleanup after tests
    index.delete(ids=["1"])  # Only delete IDs actually inserted
    pc.delete_index(INDEX_NAME)


def test_index_exists(pinecone_index):
    """Ensure the Pinecone index is created successfully."""
    assert INDEX_NAME in pc.list_indexes()


def test_upsert_and_query(pinecone_index):
    """Test inserting and retrieving a vector."""
    data = [
        {"id": "1", "chunk_text": "What is the capital of France?", "type": "instruction", "parent_id": "_"},
        {"id": "resp_1", "chunk_text": "Paris", "type": "response", "parent_id": "1"},
    ]

    # Generate vector embedding
    vector = model.encode(data[0]["chunk_text"]).tolist()
    metadata = {"response": data[1]["chunk_text"]}

    # Upsert into Pinecone
    pinecone_index.upsert(vectors=[{"id": "1", "values": vector, "metadata": metadata}])

    # Query the vector database
    query_vector = model.encode("What is the capital of France?").tolist()
    result = pinecone_index.query(vector=query_vector, top_k=1, include_metadata=True, include_values=True)

    # Ensure a result is returned
    assert result["matches"], "No results found in Pinecone"
    assert "metadata" in result["matches"][0], "Metadata missing in query result"
    assert result["matches"][0]["metadata"]["response"] == "Paris"
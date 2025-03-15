import pytest
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_2wGKDu_5hg7QhigSXtyPM7rUkjo5dYMdWtgHuwtjM3a2etqtg64rkjbc3mrCrquaEXbG4w")
# Define index name
INDEX_NAME = "test-vector-db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def pinecone_index():
    """Setup Pinecone index for testing."""
    # Create index w/ same logic as in my create_index.py script
    if not pc.has_index(INDEX_NAME):
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"chunk_text": "chunk_text"},
            },
        )

    index = pc.Index(INDEX_NAME)
    yield index  # Provide index to tests

    # Cleanup after tests
    index.delete(ids=["1", "2"])  # Delete specific test vectors
    pc.delete_index(INDEX_NAME)


def test_index_exists(pinecone_index):
    """Ensure the Pinecone index is created successfully."""
    assert INDEX_NAME in pc.list_indexes()


def test_upsert_and_query(pinecone_index):
    """Test inserting and retrieving a vector."""
    # Sample data based off created dataset
    data = [
        {
            "id": "1",
            "chunk_text": "What is the capital of France?",
            "type": "instruction",
            "parent_id": "_"
        },

        {
            "id": "resp_1",
            "chunk_text": "Paris",
            "type": "response",
            "parent_id": "1"
         }
    ]

    # Generate vector embedding
    vector = model.encode(data[0]["chunk_text"]).tolist()
    metadata = {"response": data[1]["chunk_text"]}

    # Upsert into Pinecone
    pinecone_index.upsert(vectors=[("1", vector, metadata)])

    # Query the vector database
    query_vector = model.encode("What is the capital of France").tolist()
    result = pinecone_index.query(vector=query_vector, top_k=1, include_metadata=True)

    # Ensure a result is returned
    assert result["matches"], "No results found in Pinecone"

    # Check if the response matches expected data
    assert result["matches"][0]["metadata"]["response"] == "Paris"


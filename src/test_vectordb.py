import pytest
import pinecone
from sentence_transformers import SentenceTransformer
import pytest
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_2wGKDu_5hg7QhigSXtyPM7rUkjo5dYMdWtgHuwtjM3a2etqtg64rkjbc3mrCrquaEXbG4w"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# Define index name
INDEX_NAME = "test-vector-db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def pinecone_index():
    """Setup Pinecone index for testing."""
    # Create index if not exists
    if INDEX_NAME not in pc.list_indexes():
        pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine")

    index = pc.Index(INDEX_NAME)
    yield index  # Provide index to tests

    # Cleanup after tests
    index.delete(delete_all=True)
    pc.delete_index(INDEX_NAME)


def test_index_exists(pinecone_index):
    """Ensure the Pinecone index is created successfully."""
    assert INDEX_NAME in pc.list_indexes()


def test_upsert_and_query(pinecone_index):
    """Test inserting and retrieving a vector."""
    # Sample data
    data = [
        {
            "id": "1",
            "instruction": "What is the capital of France?",
            "response": "Paris",
        }
    ]

    # Generate vector embedding
    vector = model.encode(data[0]["instruction"]).tolist()

    # Upsert into Pinecone
    pinecone_index.upsert(vectors=[("1", vector, {"response": data[0]["response"]})])

    # Query the vector database
    query_vector = model.encode("Which fruit has high potassium?").tolist()
    result = pinecone_index.query(vector=query_vector, top_k=1, include_metadata=True)

    # Ensure a result is returned
    assert result["matches"], "No results found in Pinecone"

    # Check if the response matches expected data
    assert result["matches"][0]["metadata"]["response"] == "Paris"


def test_delete_vector(pinecone_index):
    """Test deletion of a vector."""
    # Insert test vector
    test_vector = model.encode("Sample question").tolist()
    pinecone_index.upsert(vectors=[("2", test_vector, {"response": "Sample answer"})])

    # Ensure it's in the database
    result = pinecone_index.query(vector=test_vector, top_k=1, include_metadata=True)
    assert result["matches"], "Vector was not inserted correctly"

    # Delete vector
    pinecone_index.delete(ids=["2"])

    # Try querying the deleted vector
    result_after_delete = pinecone_index.query(
        vector=test_vector, top_k=1, include_metadata=True
    )

    # The response should now be empty
    assert not result_after_delete["matches"], "Vector was not deleted"


def test_empty_query(pinecone_index):
    """Ensure querying with an empty vector returns no results."""
    empty_vector = [0.0] * 384  # Vector of zeros (invalid query)
    result = pinecone_index.query(vector=empty_vector, top_k=1, include_metadata=True)

    # Should return no matches
    assert not result["matches"], "Querying with empty vector should return no results"


def test_nonexistent_id_query(pinecone_index):
    """Ensure querying for a nonexistent vector ID returns no matches."""
    test_vector = model.encode("Nonexistent question").tolist()
    result = pinecone_index.query(vector=test_vector, top_k=1, include_metadata=True)

    # Should return no matches
    assert not result["matches"], (
        "Querying a nonexistent vector should return no results"
    )

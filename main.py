import os
import time
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone import ServerlessSpec
from datasets import load_dataset, Dataset


# Mock dataset creation if real dataset fails
def create_mock_dataset():
    """Mock a simple dataset for testing."""
    data = {
        "id": [1, 2, 3],
        "text": [
            "What is the capital of France?",
            "Who is the president of the United States?",
            "What is the boiling point of water?",
        ],
    }
    return Dataset.from_dict(data)


# Try to load the actual dataset or fall back to a mock dataset
try:
    dataset = load_dataset("squad-text-embedding-ada-002")
    dataset["documents"].drop(["sparse_values", "blob"], axis=1, inplace=True)
except Exception as e:
    print(f"Failed to load dataset: {e}. Using mock dataset instead.")
    dataset = create_mock_dataset()

# Authenticate Pinecone if necessary
if not os.environ.get("PINECONE_API_KEY"):
    from pinecone_notebooks.colab import Authenticate

    Authenticate()

# Get the API key from environment variables
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise EnvironmentError("PINECONE_API_KEY is required")

# Configure the Pinecone client
pc = Pinecone(api_key=api_key)

# Set cloud and region options
cloud = os.environ.get("PINECONE_CLOUD") or "aws"
region = os.environ.get("PINECONE_REGION") or "us-east-1"

# Define the serverless spec for the index
spec = ServerlessSpec(cloud=cloud, region=region)

# Set the index name
index_name = "haozhehw7"

# If the index exists, delete it
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Create a new index
pc.create_index(
    index_name,
    dimension=1536,  # Dimensionality of text-embedding-ada-002
    metric="cosine",
    spec=spec,
)

# Wait for the index to be initialized
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Upsert the data from the dataset to the index
for batch in tqdm(
    dataset["documents"].iter_documents(batch_size=100),
    total=len(dataset["documents"]) // 100,
):
    index.upsert(batch)

# Load the model for querying
model_name = "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"  # This model outputs 1536-dimensional embeddings
model = SentenceTransformer(model_name)

# Example query
query_text = "Who is the current US president?"

# Generate the embedding for the query
xq = model.encode(query_text).tolist()

# Query the Pinecone index
xc = index.query(vector=xq, top_k=3, include_metadata=True)

# Print the query results
print(xc)

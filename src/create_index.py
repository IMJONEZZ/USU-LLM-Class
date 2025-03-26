from pinecone import Pinecone
from SampleDataset import SampleDataset
import time

# Initialize Pinecone API
pc = Pinecone(
    api_key=""
)

index_name = "first-index"

# Check if index exists, create if not
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"chunk_text": "chunk_text"},
        },
    )

# Connect to the index
first_index = pc.Index(index_name)

# Load dataset
data = SampleDataset()

# Prepare chunked records for Pinecone
chunked_records = []
for record in data.records:
    chunked_records.append(
        {
            "id": f"{record['id']}_instr",
            "chunk_text": record["chunk_text"],  # Ensures chunk_text is a string
            "type": "instruction",
            "parent_id": record["id"],
        }
    )

    chunked_records.append(
        {
            "id": f"{record['id']}_resp",
            "chunk_text": record["chunk_text"],  # Ensures chunk_text is a string
            "type": "response",
            "parent_id": record["id"],
        }
    )

# Upsert data into Pinecone
first_index.upsert_records("test-namespace", chunked_records)

# Wait for indexing
time.sleep(10)

# Check index stats
stats = first_index.describe_index_stats()
print(stats)

# Conduct semantic search
query = input("Enter a query: ")

# Search the index
results = first_index.search(
    namespace="test-namespace", query={"top_k": 10, "inputs": {"text": query}}
)

# Print search results
for hit in results["result"]["hits"]:
    print(
        f"ID: {hit['_id']}, Score: {round(hit['_score'], 2)}, Text: {hit['fields']['chunk_text']}, Type: {hit['fields']['type']}"
    )

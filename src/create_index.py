from pinecone import Pinecone
from SampleDataset import SampleDataset

pc = Pinecone(api_key="pcsk_2wGKDu_5hg7QhigSXtyPM7rUkjo5dYMdWtgHuwtjM3a2etqtg64rkjbc3mrCrquaEXbG4w")

index_name = "first-index"
if not pc.has_index(index_name):
   pc.create_index_for_model(
      name=index_name,
      cloud="aws",
      region="us-east-1",
      embed={
          "model":"llama-text-embed-v2",
          "field_map":{"text": "chunk_text"}
      }
    )

first_index = pc.Index(index_name)

data = SampleDataset()

first_index.upsert_records("test-namespace", data.records)

#wait for upserted vectors to be indexed

import time
time.sleep(10)

stats = first_index.describe_index_stats()
print(stats)

#conduct semantic search

query = input("enter in a query: ")

#search index
results = first_index.search(
    namespace="test-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text' : query
        }
    }
)

#print results
for hit in results['result']['hits']:
    print(f"id: {hit['_id']}, score : {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")

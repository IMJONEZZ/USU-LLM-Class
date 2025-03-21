from pinecone import Pinecone, ServerlessSpec
from wiki_data_digestion import WikiDataIngestion
import os

api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key)

if __name__ == "__main__":
    index_name = "wikipedia-index"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric="cosine",
            dimension=384,  # 384 dim of all-MiniLM-L6-v2
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    wiki_data_ingestion = WikiDataIngestion(index)
    wiki_data_ingestion.batch_upload()
    print(index.describe_index_stats())

    query = "Did Johannes Gutenberg invent the printing press?"
    embeddings = wiki_data_ingestion.embedder.encode(query).tolist()
    results = index.query(vector=embeddings, top_k=3, include_metadata=True)
    print(results)

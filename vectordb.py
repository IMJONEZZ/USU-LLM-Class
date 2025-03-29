from zenml import step
import faiss
import torch
from transformers import AutoModel


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to(device)
model.eval()


@step
def vectordb(dataset_embeddings, query_embeddings):
    """
    Creates a FAISS vector database from dataset embeddings,
    performs a query using query embeddings, and prints the results.
    """
    # Example dataset for reference (this might be different from your main dataset).
    dataset = [
        {"Line": "Hello world"},
        {"Line": "Hi there"},
        {"Line": "Greetings"},
        {"Line": "Goodbye"},
        {"Line": "Farewell"},
    ]

    # Determine the embedding dimension.
    d = dataset_embeddings.shape[1]

    # Create a FAISS index (using L2 distance).
    index = faiss.IndexFlatL2(d)

    # Add dataset embeddings to the index.
    index.add(dataset_embeddings)

    # For a query, we assume query_embeddings is already in the shape (n, d).
    # Here we query for the top 3 nearest neighbors.
    k = 3
    distances, indices = index.search(query_embeddings, k)

    print("Query Embedding:", query_embeddings)
    print("Nearest Neighbors' Indices:", indices)
    print("Distances:", distances)

    # Display the corresponding texts from the dataset.
    for idx in indices[0]:
        print("Matched Text:", dataset[idx]["Line"])


@step
def get_embeddings(encoding):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move inputs to the device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().astype("float32")

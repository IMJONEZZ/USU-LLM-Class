from zenml import step
from transformers import AutoTokenizer



@step
def tokenize_text(data: list):
    """Tokenizes text using Llama's tokenizer and returns tokenized tensors."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    texts = [entry["Line"] for entry in data]
    encoding = tokenizer(texts)
    return encoding

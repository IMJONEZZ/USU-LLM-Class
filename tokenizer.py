from zenml import step
from transformers import AutoTokenizer


@step
def tokenize_text(data: list):
    """
    Tokenizes text using Llama's tokenizer and returns tokenized tensors.
    Assumes each item in data is a dict with a "Line" key.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    texts = [entry["Line"] for entry in data]
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return encoding

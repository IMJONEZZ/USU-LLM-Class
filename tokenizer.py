from zenml import step
from transformers import BertTokenizerFast


@step
def tokenize_text(data: list):
    """Tokenizes text using BERT's tokenizer and returns tokenized tensors."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    texts = [entry["Line"] for entry in data]
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return encoding

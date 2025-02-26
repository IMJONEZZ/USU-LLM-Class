from transformers import AutoTokenizer
from typing import List
from zenml import step

# Load Llama tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@step
def encode_text(text: str) -> List[List[int]]:
    """Tokenizes text into subword tokens using Llama's tokenizer."""
    sentences = text.split(".")
    tokenized = [
        tokenizer.encode(sentence, add_special_tokens=True)
        for sentence in sentences
        if sentence.strip()
    ]
    return [seq for seq in tokenized if len(seq) > 1]

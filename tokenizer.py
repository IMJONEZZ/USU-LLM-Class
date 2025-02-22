from transformers import BertTokenizerFast
from typing import List
from zenml import step

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


@step
def encode_text(text: str) -> List[List[int]]:
    """Tokenizes text into subword tokens using BERT's tokenizer."""
    sentences = text.split(".")
    tokenized = [
        tokenizer.encode(sentence, add_special_tokens=True)
        for sentence in sentences
        if sentence.strip()
    ]
    return [seq for seq in tokenized if len(seq) > 1]

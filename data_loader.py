import json
from zenml import step


@step
def load_data() -> str:
    """Loads the JSON file and concatenates lines into a single large text."""
    with open("SW_EpisodeIV_VI.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [item["Line"] for item in data if "Line" in item]
    return " ".join(lines)


@step
def build_vocab(text: str) -> dict:
    """Splits text, collects all unique tokens, and assigns each a unique ID."""
    tokens = text.split()
    vocab = {"<UNK>": 0}
    vocab.update({token: i for i, token in enumerate(sorted(set(tokens)), start=1)})
    return vocab

import re
from zenml import pipeline, step
from datasets import load_dataset
from collections import Counter


class ImprovedTokenizer:
    """Tokenizer with preprocessing for lowercasing and contraction handling."""

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def preprocess(self, text: str) -> str:
        """Lowercase text and expand common contractions."""
        text = text.lower()
        contraction_dict = {
            'm': 'am', 's': 'is', 't': 'not', 're': 'are',
            've': 'have', 'll': 'will', 'd': 'would',
            'N': 'not', 'n': 'not', 'c': 'can'
        }

        def replace_contractions(match: re.Match) -> str:
            word, contraction = match.group(1), match.group(2)
            return f"{word} {contraction_dict.get(contraction, contraction)}"

        return re.sub(r"([a-zA-Z])'(m|s|t|re|ve|ll|d|N|n|c)(?=\s|$)", replace_contractions, text)

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        text = self.preprocess(text)
        tokens = [
            token.strip()
            for token in re.split(r'([,.:;?_!"()\']|--|\s)', text) if token.strip()
        ]
        tokens = [token if token in self.str_to_int else "<|unk|>" for token in tokens]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)


@step
def load_and_tokenize_data() -> dict:
    """Load data, create vocabulary, and tokenize."""
    dataset = load_dataset(
        "andrewkroening/Star-wars-scripts-dialogue-IV-VI",
        split="train[:10%]"
    )

    vocab_list = [
        token.strip()
        for line in dataset["Line"]
        for token in re.split(r'([,.:;?_!"()\']|--|\s)', line)
        if token.strip()
    ]
    word_counts = Counter(vocab_list)
    vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
    tokenizer = ImprovedTokenizer(vocab)

    first_line = dataset["Line"][0]
    print(f"Original: {first_line}")
    encoded = tokenizer.encode(first_line)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    return {'features': dataset["Line"], 'labels': dataset["Character"]}


@step
def train_model(data: dict) -> None:
    """Train the model using the dataset."""
    print(f"Trained model using {len(data['features'])} data points.")
    print(f"Label count: {len(data['labels'])}")


@pipeline
def my_pipeline_with_tokenization():
    """Define a pipeline that connects the steps."""
    dataset = load_and_tokenize_data()
    train_model(dataset)


if __name__ == "__main__":
    my_pipeline_with_tokenization()

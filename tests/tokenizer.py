import re
from zenml import pipeline, step
from datasets import load_dataset
from collections import Counter
import torch

from dataloader import StarWarsDataset, create_dataloader


class ImprovedTokenizer:
    """Tokenizer with preprocessing for lowercasing and contraction handling."""

    def __init__(self, vocab: dict, special_tokens: dict=None, max_length: int=24):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

        # Initializer special tokens
        self.special_tokens = special_tokens or {}
        self._add_special_tokens(self.special_tokens)

        #sets max sequence length
        self.max_length = max_length

    def _add_special_tokens(self, tokens: dict):
        # add special tokens to vocab if provided
        for token_name, token_symbol, in tokens.items():
            new_id = len(self.str_to_int) # assign next available ID
            self.str_to_int[token_symbol] = new_id
            self.int_to_str[new_id] = token_symbol

    def add_special_tokens(self, tokens: dict):
        #public method to add special tokens dynamically
        self._add_special_tokens(tokens)

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
        tokens = [token if token in self.str_to_int else self.special_tokens.get("unk_token", "<|unk|>") for token in tokens]

        token_ids = [self.str_to_int[token] for token in tokens]

        #Pad the sequence if it's shorter than max_length
        if len(token_ids) < self.max_length:
            token_ids += [self.str_to_int.get(self.special_tokens["pad_token"], "<|pad|>")] * (self.max_length - len(token_ids))

        #truncate the sequence if it's longer than max_length

        return token_ids[:self.max_length]


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
    special_tokens = {"unk_token": "<|unk|>", "pad_token": "<|pad|>", "cls_token": "<|cls|>"}

    tokenizer = ImprovedTokenizer(vocab, special_tokens, max_length=24)



    first_line = dataset["Line"][0]
    print(f"Original: {first_line}")
    encoded = tokenizer.encode(first_line)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    # Create PyTorch dataset & DataLoader
    torch.manual_seed(123)
    tokenizer.add_special_tokens({"unk_token": "<|unk|>"})
    star_wars_dataset = StarWarsDataset(dataset, tokenizer)
    dataloader = create_dataloader(star_wars_dataset, batch_size=10)

    # tokenize entire dataset with padding
    inputs = [tokenizer.encode(line) for line in dataset["Line"]]
    print(len(inputs))
    print(type(inputs[0][0]))
    print(type(dataset["Character"][0]))
    return {'inputs': inputs, 'labels': dataset["Character"]}


@step
def train_model(data: dict) -> None:
    """Train the model using the dataset."""
    from torch.utils.data import DataLoader, TensorDataset

    """Train the model using the dataset."""

    # Create a mapping from character names (strings) to integers
    label_to_int = {label: idx for idx, label in enumerate(set(data["labels"]))}

    # Convert the string labels to integer labels
    integer_labels = [label_to_int[label] for label in data["labels"]]

    # Ensure inputs is a tensor of shape (batch_size, sequence_length)
    inputs = torch.tensor(data["inputs"], dtype=torch.long)

    # Get the max sequence length in inputs (for padding)
    max_len = max(len(seq) for seq in data["inputs"])

    # Pad the inputs to the max length
    padded_inputs = [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in
                     data["inputs"]]

    # Convert integer labels to tensor
    labels = torch.tensor(integer_labels, dtype=torch.long)

    # Create dataset and dataloader
    dataset = TensorDataset(torch.tensor(padded_inputs, dtype=torch.long), labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Iterate through the batches
    for batch_idx, (inputs_batch, labels_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Inputs shape: {inputs_batch.shape}, Labels shape: {labels_batch.shape}")
@pipeline
def my_pipeline_with_tokenization():
    """Define a pipeline that connects the steps."""
    dataset = load_and_tokenize_data()
    train_model(dataset)


if __name__ == "__main__":
    my_pipeline_with_tokenization()

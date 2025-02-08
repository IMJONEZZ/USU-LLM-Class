import re
from collections import Counter
from tabnanny import check

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, DatasetDict
from torch.utils.data import TensorDataset, DataLoader

from zenml import pipeline, step


class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average over the sequence length
        x = self.fc(x)
        return x


class ImprovedTokenizer:
    # Tokenizer with preprocessing for lowercasing and contraction handling

    def __init__(
        self, vocab: dict, special_tokens: dict | None = None, max_length: int = 24
    ):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.special_tokens = special_tokens or {}
        self._add_special_tokens(self.special_tokens)
        self.max_length = max_length  # Sets max sequence length

    def _add_special_tokens(self, tokens: dict) -> None:
        for token_symbol in tokens.values():
            if token_symbol not in self.str_to_int:
                new_id = len(self.str_to_int)  # Assign next available ID
                self.str_to_int[token_symbol] = new_id
                self.int_to_str[new_id] = token_symbol

    def add_special_tokens(self, tokens: dict) -> None:
        # Public method to add special tokens dynamically.
        self._add_special_tokens(tokens)

    def preprocess(self, text: str) -> str:
        # Lowercase text and expand common contractions
        text = text.lower()
        contraction_dict = {
            "m": "am",
            "s": "is",
            "t": "not",
            "re": "are",
            "ve": "have",
            "ll": "will",
            "d": "would",
            "N": "not",
            "n": "not",
            "c": "can",
        }

        def replace_contractions(match: re.Match) -> str:
            word, contraction = match.group(1), match.group(2)
            return f"{word} {contraction_dict.get(contraction, contraction)}"

        return re.sub(
            r"([a-zA-Z])'(m|s|t|re|ve|ll|d|N|n|c)(?=\s|$)", replace_contractions, text
        )

    def encode(self, text: str) -> list[int]:
        # Encode text into token IDs
        text = self.preprocess(text)
        tokens = [
            token.strip()
            for token in re.split(r"([,.:;?_!\"()']|--|\s)", text)
            if token.strip()
        ]
        tokens = [
            token
            if token in self.str_to_int
            else self.special_tokens.get("unk_token", "<|unk|>")
            for token in tokens
        ]
        token_ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in tokens
        ]

        # Pad or truncate the sequence
        return token_ids[: self.max_length] + [
            self.str_to_int.get(self.special_tokens["pad_token"], 0)
        ] * max(0, self.max_length - len(token_ids))

    def decode(self, ids: list[int]) -> str:
        # Decode token IDs back into text
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r"\s+([,.:;?\"()'])", r"\1", text)


@step
def tokenize_data() -> dict:
    # Load data
    dataset = load_dataset(
        "andrewkroening/Star-wars-scripts-dialogue-IV-VI",
        split="train[:10%]",
    )

    # Create label mapping using the full dataset
    unique_labels = list(set(dataset["Character"]))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Split the dataset into training, validation, and testing sets
    dataset = dataset.train_test_split(
        test_size=0.2, seed=42
    )  # 80% training, 20% testing
    train_test_valid = dataset["train"].train_test_split(
        test_size=0.25, seed=42
    )  # 60% training, 20% validation, 20% testing

    dataset = DatasetDict(
        {
            "train": train_test_valid["train"],
            "validation": train_test_valid["test"],
            "test": dataset["test"],
        }
    )

    # Create vocabulary and tokenizer
    vocab_list = [
        token.strip()
        for line in dataset["train"]["Line"]
        for token in re.split(r"([,.:;?_!\"()']|--|\s)", line)
        if token.strip()
    ]
    word_counts = Counter(vocab_list)
    vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
    special_tokens = {
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "cls_token": "<|cls|>",
    }

    tokenizer = ImprovedTokenizer(vocab, special_tokens, max_length=24)
    vocab_size = len(tokenizer.str_to_int)

    # Tokenize the datasets
    def tokenize_dataset(dataset):
        inputs = [tokenizer.encode(line) for line in dataset["Line"]]
        max_length = 24
        padded_inputs = [
            seq[:max_length] + [0] * (max_length - len(seq)) for seq in inputs
        ]
        labels = [label_to_int[label] for label in dataset["Character"]]
        return {"inputs": padded_inputs, "labels": labels}

    train_data = tokenize_dataset(dataset["train"])
    validation_data = tokenize_dataset(dataset["validation"])
    test_data = tokenize_dataset(dataset["test"])

    return {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
        "vocab size": vocab_size,
        "label_to_int": label_to_int,  # Return the label mapping
    }


@step
def train_model(data: dict) -> str:
    train_inputs = torch.tensor(data["train"]["inputs"], dtype=torch.long)
    train_labels = torch.tensor(data["train"]["labels"], dtype=torch.long)

    validation_inputs = torch.tensor(data["validation"]["inputs"], dtype=torch.long)
    validation_labels = torch.tensor(data["validation"]["labels"], dtype=torch.long)

    train_dataset = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)

    validation_dataset = TensorDataset(validation_inputs, validation_labels)
    validation_dataloader = DataLoader(validation_dataset, batch_size=24, shuffle=False)

    # Model, Loss, Optimizer
    vocab_size = data["vocab size"]
    embed_dim = 64  # Example embedding dimension
    num_classes = len(
        data["label_to_int"]
    )  # Use the number of classes from the label mapping
    model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in validation_dataloader:
                outputs = model(batch_inputs)
                val_loss += criterion(outputs, batch_labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        val_loss /= len(validation_dataloader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy:.2f}%"
        )

    # Save the model and label mapping
    model_path = "model_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_int": data["label_to_int"],  # Save the label mapping
        },
        model_path,
    )

    return model_path  # Return the saved model path


@step
def evaluate_model(data: dict, model_path: str) -> None:
    # Load the saved model and label mapping
    checkpoint = torch.load(model_path)
    label_to_int = checkpoint["label_to_int"]

    test_inputs = torch.tensor(data["test"]["inputs"], dtype=torch.long)
    test_labels = torch.tensor(data["test"]["labels"], dtype=torch.long)

    test_dataset = TensorDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    # Define the model with the correct number of classes
    vocab_size = data["vocab size"]
    embed_dim = 64
    num_classes = len(label_to_int)  # Use the number of classes from the saved mapping
    model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)

    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_dataloader:
            outputs = model(batch_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")


@pipeline
def feature_engineering_pipeline() -> None:
    # Define a pipeline that connects the steps.
    data = tokenize_data()
    model_path = train_model(data)  # Train the model and get it
    evaluate_model(data, model_path)  # Pass the model path to evaluation


if __name__ == "__main__":
    feature_engineering_pipeline()

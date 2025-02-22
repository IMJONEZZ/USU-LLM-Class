import torch
from torch.utils.data import DataLoader, TensorDataset

from nn import SimpleTextClassifier


def evaluate_model_(data: dict, model_path: str) -> None:
    # Load the saved model and label mapping
    checkpoint = torch.load(model_path, weights_only=True)
    label_to_int = checkpoint["label_to_int"]

    test_inputs = torch.tensor(data["test"]["inputs"], dtype=torch.long)
    test_labels = torch.tensor(data["test"]["labels"], dtype=torch.long)

    test_dataset = TensorDataset(test_inputs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    # Define the model with the correct number of classes
    vocab_size = data["vocab_size"]
    embed_dim = 64
    num_classes = len(label_to_int)  # Use the number of classes from the saved mapping
    model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)

    # Load the model state
    model.load_state_dict(torch.load("best_model.pth"))
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
    return model


def predict(model, tokenizer, input_text, device="cpu"):
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Tokenize input
    tokenized_sequence = tokenizer.encode(
        input_text
    )  # Ensure this returns a list of token IDs

    # Convert to tensor and add batch dimension
    inputs = torch.tensor([tokenized_sequence]).to(device)  # Model expects a tensor

    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)  # Pass tensor directly, not as a dictionary

    # Get predicted label
    predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class  # Adjust based on your label mapping

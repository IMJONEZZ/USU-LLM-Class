import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import hashlib
from collections import Counter
from nn import SimpleTextClassifier


# Function to set deterministic behavior
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_model_hash(filepath):
    """Compute SHA256 hash of the saved model file"""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_model_(data: dict) -> str:
    set_seed(42)  # Ensure deterministic behavior

    train_inputs = torch.tensor(data["train"]["inputs"], dtype=torch.long)
    train_labels = torch.tensor(data["train"]["labels"], dtype=torch.long)

    validation_inputs = torch.tensor(data["validation"]["inputs"], dtype=torch.long)
    validation_labels = torch.tensor(data["validation"]["labels"], dtype=torch.long)

    train_dataset = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)

    validation_dataset = TensorDataset(validation_inputs, validation_labels)
    validation_dataloader = DataLoader(validation_dataset, batch_size=24, shuffle=False)

    # Model, Loss, Optimizer
    vocab_size = data["vocab_size"]
    embed_dim = 64  # Example embedding dimension
    num_classes = len(
        data["label_to_int"]
    )  # Use the number of classes from the label mapping
    model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)

    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure all classes in label_to_int are represented
    num_classes = len(data["label_to_int"])  # Should be 22
    label_counts = Counter(data["train"]["labels"])
    total_samples = sum(label_counts.values())

    # Compute weights for all classes, assigning 1 if a class is missing
    class_weights = [
        total_samples
        / (
            num_classes * (label_counts.get(i, 1))
        )  # Use .get(i, 1) to avoid zero division
        for i in range(num_classes)
    ]

    # Convert to tensor and move to device
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_inputs, batch_labels in train_dataloader:
            batch_inputs, batch_labels = (
                batch_inputs.to(device),
                batch_labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in validation_dataloader:
                batch_inputs, batch_labels = (
                    batch_inputs.to(device),
                    batch_labels.to(device),
                )

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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved")

    # Check model hash consistency before final saving
    initial_hash = get_model_hash("best_model.pth")

    # Save the model and label mapping
    model_path = "model_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_int": data["label_to_int"],  # Save the label mapping
        },
        model_path,
    )

    final_hash = get_model_hash("best_model.pth")

    print(f"Model Hash Before Saving: {initial_hash}")
    print(f"Model Hash After Saving:  {final_hash}")

    # Check if hashes match
    assert initial_hash == final_hash, "ERROR: Model file changed unexpectedly!"

    return model_path  # Return the saved model path

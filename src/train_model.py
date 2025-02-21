import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from collections import Counter

from nn import SimpleTextClassifier


def train_model_(data: dict) -> str:
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

    # Ensure all classes in label_to_int are represented
    num_classes = len(data["label_to_int"])  # Should be 22
    label_counts = Counter(data["train"]["labels"])
    total_samples = sum(label_counts.values())

    # Compute weights for all classes, assigning 1 if a class is missing
    class_weights = [
        total_samples / (num_classes * (label_counts.get(i, 1)))  # Use .get(i, 1) to avoid zero division
        for i in range(num_classes)
    ]

    # Convert to tensor and move to device
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to("cuda" if torch.cuda.is_available() else "cpu"))


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 10

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for batch_inputs, batch_labels in train_dataloader:
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

        #save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved")

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # ✅ Using PyTorch's AdamW
from transformers import BertForMaskedLM
from dataset import NextTokenDataset, collate_fn
from zenml import step

@step
def train_model(data_splits: dict, batch_size=8, epochs=2, model_name="bert-base-uncased") -> dict:
    """Trains a BERT-based model for masked language modeling."""

    # Detect device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained BERT model
    model = BertForMaskedLM.from_pretrained(model_name).to(device)

    train_dataset = NextTokenDataset(data_splits["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Move data to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Generate attention_mask where 0 is padding, and 1 is actual data
            attention_mask = (x_batch != 0).long()

            outputs = model(input_ids=x_batch, attention_mask=attention_mask, labels=y_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

    return {"train_loss": avg_train_loss}

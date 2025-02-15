import json
from typing import Optional, List, Dict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from zenml import pipeline, step
import torch.nn.functional as F 

# --------------------------------------------------
# THE SIMPLETOKENIZER CLASS
# --------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab: dict, merges: Optional[list[tuple[str, str]]] = None):
        if merges is None:
            merges = []
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int.get(token, 0) for token in text.split()]  # 0 is <UNK>

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.int_to_str.get(i, "<UNK>") for i in ids])

# --------------------------------------------------
# 1) LOAD DATA
# --------------------------------------------------
@step
def load_data() -> str:
    """
    Loads the JSON file and concatenates lines into a single large text.
    """
    with open("SW_EpisodeIV_VI.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = [item["Line"] for item in data if "Line" in item]
    return " ".join(lines)

# --------------------------------------------------
# 2) BUILD VOCABULARY
# --------------------------------------------------
@step
def build_vocab(text: str) -> dict:
    """
    Splits text, collects all unique tokens, and assigns each a unique ID (int).
    """
    tokens = text.split()
    vocab = {"<UNK>": 0}  # Add <UNK> token as ID 0
    vocab.update({token: i for i, token in enumerate(sorted(set(tokens)), start=1)})
    return vocab

# --------------------------------------------------
# 3) ENCODE TEXT
# --------------------------------------------------
@step
def encode_text(text: str, vocab: dict) -> List[List[int]]:
    """
    Splits text into sentences (naively by '.'), then tokenizes each sentence.
    """
    sentences = text.split(".")
    tokenizer = SimpleTokenizer(vocab=vocab)
    tokenized = [tokenizer.encode(sentence) for sentence in sentences if sentence.strip()]
    # Remove any empty after tokenization
    return [seq for seq in tokenized if len(seq) > 1]

# --------------------------------------------------
# SPLIT DATA INTO TRAIN, VAL, TEST
# --------------------------------------------------
@step
def split_data(tokenized_sequences: List[List[int]], 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15) -> Dict[str, List[List[int]]]:
    """
    Splits tokenized sequences into train, val, and test sets.
    """
    total = len(tokenized_sequences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = tokenized_sequences[:train_end]
    val_data = tokenized_sequences[train_end:val_end]
    test_data = tokenized_sequences[val_end:]

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

# --------------------------------------------------
# CUSTOM DATASET CLASS (for next-token prediction)
# --------------------------------------------------
class NextTokenDataset(Dataset):
    """
    Expects tokenized sequences. For each sequence, we create pairs (x, y):
      x: all but the last token
      y: all but the first token  (the "next token" for each position)
    """
    def __init__(self, tokenized_sequences: List[List[int]]):
        self.x_data = []
        self.y_data = []
        for seq in tokenized_sequences:
            # e.g., if seq = [10, 22, 15], x=[10, 22], y=[22, 15]
            # We'll only keep sequences of length > 1
            x = seq[:-1]
            y = seq[1:]
            self.x_data.append(torch.tensor(x, dtype=torch.long))
            self.y_data.append(torch.tensor(y, dtype=torch.long))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# --------------------------------------------------
# COLLATE FUNCTION FOR PADDING
# --------------------------------------------------
def collate_fn(batch, max_len=50):
    """
    Pads x and y to a fixed max length (50).
    Returns:
        padded_x, padded_y
    """
    xs, ys = zip(*batch)  # separate x and y

    padded_x = pad_sequence(xs, batch_first=True, padding_value=0)
    padded_y = pad_sequence(ys, batch_first=True, padding_value=-100)  # -100 is ignored_idx in CrossEntropy

    # If needed, ensure every sequence is exactly max_len
    # We'll do it for both x and y
    if padded_x.shape[1] < max_len:
        pad_amount = max_len - padded_x.shape[1]
        padded_x = torch.cat([
            padded_x, torch.zeros((padded_x.shape[0], pad_amount), dtype=torch.long)
        ], dim=1)
    else:
        padded_x = padded_x[:, :max_len]

    if padded_y.shape[1] < max_len:
        pad_amount = max_len - padded_y.shape[1]
        padded_y = torch.cat([
            padded_y, torch.full((padded_y.shape[0], pad_amount), -100, dtype=torch.long)
        ], dim=1)
    else:
        padded_y = padded_y[:, :max_len]

    return padded_x, padded_y

# --------------------------------------------------
# SIMPLE MODEL FOR NEXT-TOKEN PREDICTION
# --------------------------------------------------
class SimpleLanguageModel(nn.Module):
    """
    A minimal model that takes token IDs -> embeddings -> linear -> predicts next token IDs.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len] of token IDs
        returns: [batch_size, seq_len, vocab_size]
        """
        embeddings = self.embed(x)                # [batch, seq_len, embed_dim]
        logits = self.fc(embeddings)              # [batch, seq_len, vocab_size]
        return logits

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> float:
    """
    Runs inference on the dataloader and calculates avg loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss

# --------------------------------------------------
# GATHER DETAILED EVAL INFO FOR "SMART LEARNING"
# --------------------------------------------------
def gather_detailed_eval_info(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> List[Dict]:
    """
    Iterates over the dataloader, records per-token predictions vs.
    ground truth, plus the model's confidence in the correct token.
    This info can guide future "smart learning" (e.g., focusing on
    low-confidence or incorrect tokens).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)  # shape [B, seq_len, vocab_size]
            # Softmax across vocab dimension to get probabilities
            probs = F.softmax(logits, dim=-1)  # shape [B, seq_len, vocab_size]

            # Argmax for predicted tokens
            predictions = torch.argmax(logits, dim=-1)

            # We'll walk each sample in the batch
            for b_idx in range(x_batch.size(0)):
                # Gather token-level detail
                sample_info = []
                for t_idx in range(x_batch.size(1)):
                    true_token = y_batch[b_idx, t_idx].item()
                    if true_token == -100:
                        # Past the real sequence
                        break

                    pred_token = predictions[b_idx, t_idx].item()
                    confidence_correct = probs[b_idx, t_idx, true_token].item()
                    is_correct = (pred_token == true_token)

                    sample_info.append({
                        "true_token_id": true_token,
                        "pred_token_id": pred_token,
                        "confidence_in_true_token": confidence_correct,
                        "is_correct": is_correct
                    })

                results.append(sample_info)

    return results

# --------------------------------------------------
# TRAINING STEP
# --------------------------------------------------
@step
def train_model(
    data_splits: Dict[str, List[List[int]]],
    batch_size: int = 8,
    num_workers: int = 0,   # 0 if on Windows
    epochs: int = 2,
    embed_dim: int = 32
) -> Dict[str, str]:
    """
    Trains a simple language model for next-token prediction.
    Also runs validation after each epoch to show performance,
    then tests the final model on the test set. 
    Finally, gathers additional evaluation output that can
    aid in future "smart learning."
    """
    # --------------------------------------------------
    # Create Datasets
    # --------------------------------------------------
    train_dataset = NextTokenDataset(data_splits["train"])
    val_dataset   = NextTokenDataset(data_splits["val"])
    test_dataset  = NextTokenDataset(data_splits["test"])

    # We'll figure out vocab_size from max token ID in all sets
    all_seqs = data_splits["train"] + data_splits["val"] + data_splits["test"]
    max_token_id = max((max(seq) for seq in all_seqs), default=0)
    vocab_size = max_token_id + 1  # +1 because IDs are 0-indexed

    # --------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # --------------------------------------------------
    # Initialize Model, Loss, Optim
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)  # [batch, seq_len, vocab_size]
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # --------------------------------------------------
    # Final Test Evaluation
    # --------------------------------------------------
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # --------------------------------------------------
    # Gather details for future "smart learning"
    # --------------------------------------------------
    detailed_test_info = gather_detailed_eval_info(model, test_loader, device)

    # We can print the length or a small snippet to show it's working
    # (In a real scenario, you might save it to a JSON file or a DB.)
    print(f"Gathered test info items: {len(detailed_test_info)} sequences total.")
    if detailed_test_info:
        # Print just the first item for demonstration
        print("Example of per-token test info:")
        print(detailed_test_info[0])  # A single sequence's data

    return {
        "train_loss": f"{avg_train_loss:.4f}",
        "val_loss": f"{val_loss:.4f}",
        "test_loss": f"{test_loss:.4f}",
        "detail_count": f"{len(detailed_test_info)}"
    }

# --------------------------------------------------
# PIPELINE DEFINITION
# --------------------------------------------------
@pipeline
def train_eval_pipeline():
    """
    Full pipeline:
      1) load data
      2) build vocab
      3) encode text
      4) split data
      5) train model (includes validation + test)
         plus gather detailed outputs for future "smart learning."
    """
    text = load_data()
    vocab = build_vocab(text)
    token_ids = encode_text(text, vocab)
    data_splits = split_data(token_ids)
    train_model(data_splits)

# --------------------------------------------------
# RUN THE PIPELINE
# --------------------------------------------------
if __name__ == "__main__":
    train_eval_pipeline()

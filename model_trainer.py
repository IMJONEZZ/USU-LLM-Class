from typing import Dict, Optional
import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertGenerationConfig,
    BertGenerationDecoder,
)
from transformers import get_linear_schedule_with_warmup
from zenml import step, log_metadata
from zenml.logger import get_logger

logger = get_logger(__name__)


class StarWarsDataset(Dataset):
    """Custom dataset for Star Wars dialogue data."""

    def __init__(self, df: pd.DataFrame, max_length: int = 64):
        self.texts = df["tokens"].values
        self.max_length = max_length

        logger.info("Validating sequences...")
        valid_sequences = []
        for tokens in self.texts:
            try:
                # Handle string representation of lists
                if isinstance(tokens, str):
                    try:
                        # Only eval if it looks like a list
                        if tokens.strip().startswith("[") and tokens.strip().endswith(
                            "]"
                        ):
                            tokens = eval(tokens)
                        else:
                            continue
                    except (SyntaxError, ValueError, NameError):
                        continue

                # Skip None values
                if tokens is None:
                    continue

                # Verify it's a valid sequence type
                if not isinstance(tokens, (list, np.ndarray)):
                    continue

                # Verify all elements are integers
                if not all(isinstance(t, (int, np.integer)) for t in tokens):
                    continue

                valid_sequences.append(tokens)

            except Exception as e:
                logger.debug(f"Skipping invalid sequence: {str(e)}")
                continue

        self.texts = valid_sequences
        logger.info(f"Found {len(self.texts)} valid sequences")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens)

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            pad_length = self.max_length - len(tokens)
            tokens = np.pad(tokens, (0, pad_length), "constant", constant_values=0)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = np.ones_like(tokens)
        attention_mask[tokens == 0] = 0

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@step
def bert_trainer(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    tokenizer_config: Dict,
    batch_size: int = 16,
    max_length: int = 64,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    device: Optional[str] = None,
) -> Dict:
    """Train a BERT model on Star Wars dialogue data."""
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Creating training dataset...")
        train_dataset = StarWarsDataset(train_data, max_length=max_length)
        logger.info("Creating validation dataset...")
        val_dataset = StarWarsDataset(validation_data, max_length=max_length)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("No valid sequences found in dataset")

        # Create model configuration
        vocab_size = len(tokenizer_config["str_to_int"])
        config = BertGenerationConfig(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=512,
            pad_token_id=tokenizer_config["str_to_int"].get("<|pad|>", 0),
            bos_token_id=tokenizer_config["str_to_int"].get("<|endoftext|>", 1),
            eos_token_id=tokenizer_config["str_to_int"].get("<|endoftext|>", 1),
            is_decoder=True,
            add_cross_attention=True,  # Enable cross-attention
            tie_word_embeddings=True,
        )

        # Create decoder-only model
        decoder = BertGenerationDecoder(config).to(device)

        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=learning_rate, weight_decay=0.01
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_loader) * num_epochs,
        )

        best_val_loss = float("inf")
        training_stats = []

        for epoch in range(num_epochs):
            # Training phase
            decoder.train()
            total_train_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                try:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    # Forward pass with causal masking for language modeling
                    outputs = decoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,  # Use input as labels for language modeling
                        return_dict=True,
                        use_cache=False,  # Disable caching to save memory
                    )

                    loss = outputs.loss
                    total_train_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if batch_idx % 50 == 0:
                        logger.info(
                            f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

                del outputs, loss
                if device == "cuda":
                    torch.cuda.empty_cache()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            decoder.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    try:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        outputs = decoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids,
                            return_dict=True,
                            use_cache=False,
                        )

                        total_val_loss += outputs.loss.item()

                        del outputs
                        if device == "cuda":
                            torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue

            avg_val_loss = total_val_loss / len(val_loader)

            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            training_stats.append(epoch_stats)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        log_metadata(
            metadata={
                "final_train_loss": training_stats[-1]["train_loss"],
                "final_val_loss": training_stats[-1]["val_loss"],
                "best_val_loss": best_val_loss,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        )

        model_artifact = {
            "model_state": decoder.state_dict(),
            "config": config.to_dict(),
            "training_stats": training_stats,
        }

        return model_artifact

    except Exception as e:
        logger.error(f"Error in BERT training: {str(e)}")
        raise

from zenml import step
import sys
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


# Set encoding to UTF-8 for output
sys.stdout.reconfigure(encoding="utf-8")


@step
def SFT_train(train_dataset):
    torch.cuda.empty_cache()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,  # Use half precision
    )

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        report_to="none",  # Disable Weights & Biases logging
    )

    # Create SFTTrainer instance
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack(
                [f["input_ids"] for f in data]
            ),  # Use input_ids as labels
        },
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./star_wars_llama")
    tokenizer.save_pretrained("./star_wars_llama")

    print("Model training complete and saved to './star_wars_llama'")

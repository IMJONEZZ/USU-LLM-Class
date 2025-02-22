from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import os

os.environ["WANDB_DISABLED"] = "true"


# 1. Load LLaMA-2 7B in 4-bit mode
model_name = "meta-llama/Llama-2-7b-hf"
max_seq_length = 512  # Maximum sequence length

# Load model in 4-bit quantized mode
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name, load_in_4bit=True, max_seq_length=max_seq_length
)

# 2. Attach LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # LoRA rank (controls trainable parameter count)
    lora_alpha=32,  # Scaling factor for LoRA
    lora_dropout=0.05,  # Dropout rate to prevent overfitting
    bias="none",  # No bias training
    use_rslora=False,  # Keep standard LoRA
)

# Print trainable parameters
model.print_trainable_parameters()

# 3. Load and Tokenize Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Use subset of the dataset for quicker training
small_train_dataset = tokenized_datasets["train"].select(
    range(100)
)  # Use first 100 examples
small_eval_dataset = tokenized_datasets["validation"].select(
    range(10)
)  # Use first 10 examples

# 4. Training Setup
training_args = TrainingArguments(
    output_dir="./llama-unsloth",
    per_device_train_batch_size=20,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=1,  # Train for just 1 epoch
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=True,  # Enables mixed precision training
    logging_steps=10,  # Log every 10 steps
)

# data collator for Causal LM training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False for autoregressive models like LLaMA
)


# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,  # Add this to correctly handle labels
)

# 6. Train the Model
trainer.train()


# 7. Return the trainer object for testing
def get_trainer():
    return trainer

# trained on colab

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset(
    "json", data_files="/content/drive/MyDrive/datasets/stories.json"
)

# Load tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Preprocess function to tokenize data
def preprocess(example):
    return {
        "input_ids": tokenizer(
            example["instruction"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )["input_ids"],
        "labels": tokenizer(
            example["response"], truncation=True, padding="max_length", max_length=512
        )["input_ids"],
    }


# Apply preprocessing to dataset
dataset = dataset.map(preprocess, remove_columns=["instruction", "response"])

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./story_llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    bf16=True,
    gradient_checkpointing=True,  # Enable gradient checkpointing
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])

# Train the model

torch.cuda.empty_cache()
trainer.train()

# Save model and tokenizer
model.save_pretrained("llama3_story_generator")
tokenizer.save_pretrained("llama3_story_generator")

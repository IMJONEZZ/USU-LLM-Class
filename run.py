from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

PatchFastRL("GRPO", FastLanguageModel)

from transformers import Trainer, TrainingArguments

max_seq_length = 512  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

import os

# List all files in the input directory
input_dir = "/kaggle/input/"

# Loop through the directory to print dataset names
for root, dirs, files in os.walk(input_dir):
    print(f"📂 Directory: {root}")
    for dir_name in dirs:
        print(f"   📁 Dataset: {os.path.join(root, dir_name)}")
    for file_name in files:
        print(f"   📄 File: {os.path.join(root, file_name)}")
    print("-" * 50)

import os
from datasets import load_dataset

print(os.getcwd())
file_path = "/kaggle/input/star-wars3/star-wars.json"

dataset = load_dataset("json", data_files={"train": file_path})["train"]
display(dataset)

dataset = load_dataset("json", data_files={"train": file_path})["train"]


# Preprocessing: Combine the character and line into one string.
def preprocess(example):
    # Create a dialogue-like string (e.g., "THREEPIO: We're doomed!")
    text = f"{example['Character']}: {example['Line']}"
    # Tokenize the text; we use padding and truncation to ensure uniform length.
    tokenized = tokenizer(
        text, truncation=True, max_length=max_seq_length, padding="max_length"
    )
    # For language modeling, labels are the same as the input IDs.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# Apply the preprocessing to the entire dataset.
tokenized_dataset = dataset.map(preprocess, batched=False)

display(tokenized_dataset)

# Define training arguments using Hugging Face's Trainer API.
training_args = TrainingArguments(
    output_dir="./star_wars_model",  # Directory to save model checkpoints
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    gradient_accumulation_steps=1,  # Increase for smoother training if needed
    num_train_epochs=2,  # Set epochs based on your dataset size
    learning_rate=5e-6,  # Adjust the learning rate as needed
    weight_decay=0.1,
    logging_steps=10,  # Log every 10 steps
    save_steps=500,  # Save a checkpoint every 500 steps
    fp16=not is_bfloat16_supported(),  # Use FP16 if BF16 is not supported
    bf16=is_bfloat16_supported(),
    report_to="none",
)

# Initialize the Trainer; this automatically uses cross-entropy loss for language modeling.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

from vllm import SamplingParams

text = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Tell me a bedtime story with Yoda's voice",
        },
    ],
    tokenize=False,
    add_generation_prompt=True,
)


sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0]
    .outputs[0]
    .text
)

output

model.save_pretrained("/kaggle/working/star_wars_model_finetuned")

import os

print(os.listdir("/kaggle/working/"))  # Lists files in the working directory
print(os.listdir("/kaggle/input/"))  # Lists datasets you've added to the notebook

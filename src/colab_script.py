# Install dependencies
!pip install -q unsloth vllm
!pip install -q --upgrade pillow
!pip install -q git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b

# Imports
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import Dataset
import torch
import json

# Reward function for GRPO
def reward_function(samples, scores, *kwargs):
    print("Samples:", samples)  # Debugging
    print("Scores:", scores)    # Debugging

    if not samples or not scores:
        return [0] * len(samples)  # Prevents crashes

    return [score + 0.1 for score in scores]  # Simple reward update

# Load dataset from JSONL file
dataset_path = "/content/drive/MyDrive/datasets/train.jsonl"  # Adjust path if needed

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]  # Ignore empty lines

# Load the dataset
data = load_jsonl(dataset_path)

# Convert to expected format (renaming "instruction" to "prompt")
def convert_dataset_format(data):
    return [{"prompt": item["instruction"], "response": item["response"]} for item in data]

data = convert_dataset_format(data)  # Ensure keys match expected format

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)


# Patch Unsloth for GRPO
PatchFastRL("GRPO", FastLanguageModel)

# Model Configuration
max_seq_length = 512
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # Enable quantization for memory efficiency
    fast_inference=True,  # Use vLLM for fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.4,
)

# Apply LoRA for Efficient Fine-Tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Training Configuration
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm=True,  # Enables vLLM acceleration
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=256,
    max_completion_length=200,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_function()],
)

# Start Fine-Tuning
trainer.train()

# Save Model & Tokenizer
model.save_pretrained("outputs")
tokenizer.save_pretrained("outputs")


import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# Quantization config
bnb_4bit_compute_dtype = torch.float32
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head"],
)

# Training config
training_kwargs = {
    "output_dir": "./text2cypher-lora",
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "logging_steps": 50,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "optim": "adamw_torch",
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "report_to": "none",
    "remove_unused_columns": False,
}

MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DATASET_NAME = "neo4j/text2cypher-2024v1"
MAX_SEQ_LENGTH = 512

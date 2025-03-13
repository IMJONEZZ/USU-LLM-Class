"""
Model training functionality for Llama 3.2 fine-tuning
"""

import os
import torch
from typing import Dict, Any
from typing_extensions import Annotated
from datasets import Dataset
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml import log_artifact_metadata

from config import (
    HUGGINGFACE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    NUM_TRAIN_EPOCHS,
    LEARNING_RATE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    OUTPUT_DIR,
    HF_TOKEN
)
from utils import use_unsloth

logger = get_logger(__name__)

@step
def train_llama_model(
    dataset: Dataset,
    huggingface_model_name: str = HUGGINGFACE_MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    lora_rank: int = LORA_RANK,
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    output_dir: str = OUTPUT_DIR,
    hf_token: str = HF_TOKEN
) -> Annotated[Dict[str, Any], "model_info"]:
    """Train Llama 3.2:1B model with either Unsloth (GPU) or standard transformers (CPU)."""
    # Import necessary libraries
    from transformers import TrainingArguments, Trainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Create or get the model for ZenML tracking
    client = Client()
    zenml_model_name = "Assn6LLMFinetune"

    # Check if the model exists
    try:
        # Try to get the model by name
        model = client.get_model(zenml_model_name)
        logger.info(f"Found existing model: {zenml_model_name}")
    except KeyError:
        # Model doesn't exist, create it
        logger.info(f"Creating new model: {zenml_model_name}")
        model = client.create_model(
            name=zenml_model_name,
            description="Llama 3.21B model finetuned for Assignment 6",
            tags=["llama", "instruction-tuning", "assignment"]
        )

    # Check if we should use Unsloth (GPU required)
    use_unsloth_backend = use_unsloth()
    
    # Load model and tokenizer (different method based on backend)
    logger.info(f"Loading model {huggingface_model_name}")
    
    if use_unsloth_backend:
        # Only import unsloth if we're really going to use it
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=huggingface_model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,  # Use 4-bit quantization
                token=hf_token,     # Use the HF token for authentication
            )
            
            # Apply LoRA with Unsloth
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=lora_rank,
                lora_dropout=0.05,
                use_gradient_checkpointing="unsloth",
            )
        except ImportError:
            logger.warning("Failed to import unsloth, falling back to standard Transformers")
            use_unsloth_backend = False
    
    # Use standard transformers if unsloth failed or wasn't selected
    if not use_unsloth_backend:
        # Use standard transformers for CPU
        tokenizer = AutoTokenizer.from_pretrained(
            huggingface_model_name,
            token=hf_token,
        )
        
        # Set tokenizer padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model - with lower precision for CPU
        model = AutoModelForCausalLM.from_pretrained(
            huggingface_model_name,
            token=hf_token,
            device_map="auto",   # Works for both CPU and lower-end GPUs
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
            # Skip quantization when on CPU to avoid bitsandbytes issues
            load_in_8bit=False,
            quantization_config=None
        )
        
        # Configure LORA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LORA to model
        model = get_peft_model(model, lora_config)
        
    # Log model parameters
    model_params = {
        "model_name": huggingface_model_name,
        "lora_rank": lora_rank,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "backend": "unsloth" if use_unsloth_backend else "transformers",
    }
    log_artifact_metadata(model_params)

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Prepare tokenization function
    def tokenize_function(example):
        # Format messages into conversation
        messages = example["messages"]
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

        # Tokenize input and target
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length",
                          max_length=max_seq_length, truncation=True)
        targets = tokenizer(example["target"], return_tensors="pt", padding="max_length",
                           max_length=max_seq_length, truncation=True)

        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": targets.input_ids[0]
        }

    # Process the datasets
    logger.info("Tokenizing datasets")
    tokenized_train = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_function, remove_columns=eval_dataset.column_names)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Reduce batch size for CPU training if needed
    actual_batch_size = per_device_train_batch_size
    actual_grad_accum = gradient_accumulation_steps
    
    if not torch.cuda.is_available():
        logger.warning("Running on CPU - reducing batch size and increasing gradient accumulation steps")
        actual_batch_size = 1
        actual_grad_accum = gradient_accumulation_steps * per_device_train_batch_size
        logger.info(f"New batch size: {actual_batch_size}, new grad_accum: {actual_grad_accum}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=actual_batch_size,
        gradient_accumulation_steps=actual_grad_accum,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),  # Only use fp16 if on GPU
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Don't use wandb, etc.
        max_grad_norm=1.0,
        optim="adamw_torch",  # Memory-efficient optimizer
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log final metrics
    final_metrics = {
        "train_loss": train_result.metrics.get("train_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }

    # Evaluate the model
    logger.info("Evaluating model...")
    eval_metrics = trainer.evaluate()

    # Combine metrics
    final_metrics.update({
        "eval_loss": eval_metrics.get("eval_loss", 0),
    })

    # Log final metrics
    log_artifact_metadata(final_metrics)

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "model_path": output_dir,
        "tokenizer_path": output_dir,
        "model_name": zenml_model_name,
        "backend_used": "unsloth" if use_unsloth_backend else "transformers"
    }
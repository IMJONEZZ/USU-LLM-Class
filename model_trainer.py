from typing import Dict, Any, Optional
from typing_extensions import Annotated
import os
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def train_llama_model(
    dataset: Dataset,
    huggingface_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    hf_token: Optional[str] = None,
    max_seq_length: int = 512,
    lora_rank: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    output_dir: str = "llama_finetuned",
) -> Annotated[Dict[str, Any], "model_info"]:
    """Train Llama 3.2:1B model with Unsloth.

    This step fine-tunes the LLaMA 3.2 1B model using LoRA (Low-Rank Adaptation)
    for parameter-efficient fine-tuning.

    Args:
        dataset: The training dataset
        huggingface_model_name: HuggingFace model identifier
        hf_token: HuggingFace API token for accessing the model
        max_seq_length: Maximum sequence length for training
        lora_rank: Rank of LoRA adaptation matrices
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        per_device_train_batch_size: Batch size per GPU/TPU core
        gradient_accumulation_steps: Number of steps to accumulate gradients
        output_dir: Directory to save the model

    Returns:
        Dict with model information and paths
    """
    # Import Unsloth and Accelerate
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth is not installed. Please run 'pip install unsloth'")
        raise

    # Validate HF token
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "HuggingFace token is required. Please provide it via the 'hf_token' argument "
                "or set the 'HF_TOKEN' environment variable."
            )

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Log dataset split info
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    # Create model directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer with Unsloth
    logger.info(f"Loading model {huggingface_model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=huggingface_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # Use 4-bit quantization
        token=hf_token,  # Use the HF token for authentication
    )

    # Apply LoRA
    logger.info(f"Applying LoRA with rank {lora_rank}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
    )

    # Log model parameters
    model_params = {
        "model_name": huggingface_model_name,
        "lora_rank": lora_rank,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    log_artifact_metadata(model_params)

    # Prepare tokenization function
    def tokenize_function(example):
        # Format messages into conversation
        messages = example["messages"]
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

        # Tokenize input and target
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )
        targets = tokenizer(
            example["target"],
            return_tensors="pt",
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": targets.input_ids[0],
        }

    # Process the datasets
    logger.info("Tokenizing datasets")
    tokenized_train = train_dataset.map(
        tokenize_function, remove_columns=train_dataset.column_names
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, remove_columns=eval_dataset.column_names
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Don't use wandb, etc.
        max_grad_norm=1.0,
        optim="adamw_torch",  # More memory-efficient optimizer
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
        "train_samples_per_second": train_result.metrics.get(
            "train_samples_per_second", 0
        ),
    }

    # Evaluate the model
    logger.info("Evaluating model...")
    eval_metrics = trainer.evaluate()

    # Combine metrics
    final_metrics.update(
        {
            "eval_loss": eval_metrics.get("eval_loss", 0),
        }
    )

    # Log final metrics
    log_artifact_metadata(final_metrics)

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "model_path": output_dir,
        "tokenizer_path": output_dir,
        "model_name": "Llama-3.2-1B-Finetuned",
    }


class ZenMLLoggingCallback:
    """Custom callback for ZenML logging during training."""

    def __init__(self):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    log_artifact_metadata({key: value})

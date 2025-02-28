#!/usr/bin/env python3
"""
LLaMA 3.2 1B Finetuning Pipeline for Assignment 6

This script runs the complete pipeline for finetuning the LLaMA 3.2 1B model,
from dataset preparation to model evaluation on test questions.
"""

import os
import argparse
from pathlib import Path
from zenml import pipeline
from zenml.logger import get_logger
from zenml.client import Client

# Import our components
from data_preprocessor import create_dataset, test_questions
from model_trainer import train_llama_model
from model_evaluator import test_model

# Configure logging
logger = get_logger(__name__)


@pipeline
def llama_finetuning_pipeline(
    huggingface_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    hf_token: str = None,
    max_seq_length: int = 512,
    lora_rank: int = 16,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_new_tokens: int = 256,
):
    """Pipeline for finetuning Llama 3.2:1B on instruction data.

    Args:
        huggingface_model_name: HuggingFace model identifier
        hf_token: HuggingFace API token
        max_seq_length: Maximum sequence length
        lora_rank: LoRA rank parameter
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Steps for gradient accumulation
        max_new_tokens: Maximum new tokens for generation

    Returns:
        Tuple of model info and evaluation results
    """
    # Create the dataset
    dataset = create_dataset()

    # Get the list of test questions
    questions = test_questions()

    # Train the model
    model_info = train_llama_model(
        dataset=dataset,
        huggingface_model_name=huggingface_model_name,
        hf_token=hf_token,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Test on assignment questions
    results = test_model(
        model_info=model_info,
        test_questions=questions,
        hf_token=hf_token,
        max_new_tokens=max_new_tokens,
    )

    # Return both model information and evaluation results
    return model_info, results


def main():
    """Run the LLaMA finetuning pipeline with command line arguments."""
    parser = argparse.ArgumentParser(description="Finetune LLaMA 3.2 1B model")
    parser.add_argument("--hf_token", type=str, help="HuggingFace API token")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--output_dir", type=str, default="./llama_finetuned", help="Output directory"
    )
    args = parser.parse_args()

    # Initialize ZenML if not already initialized
    try:
        client = Client()
        logger.info(f"Connected to ZenML server at {client.zen_store.url}")
    except Exception as e:
        logger.info(f"Initializing ZenML: {str(e)}")
        from zenml.cli.cli import init

        init()

    # Check if HF_TOKEN is provided or in environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        parser.error(
            "HuggingFace token is required. Provide --hf_token or set HF_TOKEN environment variable."
        )

    try:
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Run the pipeline
        logger.info("Starting LLaMA 3.2 1B finetuning pipeline...")
        pipeline_result = llama_finetuning_pipeline(
            hf_token=hf_token,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
        )

        logger.info("Pipeline execution completed successfully!")

        # Print path to answers.txt
        answers_path = Path("answers.txt").absolute()
        logger.info(f"Model answers saved to: {answers_path}")

        return pipeline_result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

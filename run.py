"""
Main script to run the Llama 3.2 fine-tuning pipeline
"""

import os
import argparse
from zenml.logger import get_logger
from utils import setup_environment, check_gpu
from pipeline import llama_finetuning_pipeline
from evaluation import show_model_answers

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Llama 3.2 fine-tuning pipeline")
    
    # Required credentials
    parser.add_argument("--zenml-url", type=str, help="ZenML server URL")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help="HuggingFace model name")
    parser.add_argument("--lora-rank", type=int, default=16, 
                        help="LoRA rank parameter")
    parser.add_argument("--seq-length", type=int, default=512, 
                        help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Per-device training batch size")
    parser.add_argument("--grad-accum", type=int, default=4, 
                        help="Gradient accumulation steps")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="llama_finetuned", 
                        help="Output directory for model")
    
    # Backend selection (force CPU if needed)
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    # Get credentials from environment if not provided
    zenml_url = args.zenml_url or os.environ.get("ZENML_SERVER_URL")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Set environment variables for later use
    if zenml_url:
        os.environ["ZENML_SERVER_URL"] = zenml_url
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Handle forced CPU usage if requested
    if args.force_cpu:
        logger.info("Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide GPU from PyTorch
        
        # Set a global flag to avoid even trying to import Unsloth
        import utils
        utils.FORCE_CPU_MODE = True
        print("CPU-only mode enabled")
    
    # Setup environment and check for GPU
    client = setup_environment(zenml_url, hf_token)
    device = check_gpu()
    
    # Run the pipeline
    logger.info("Starting Llama 3.2 fine-tuning pipeline")
    model_info, results = llama_finetuning_pipeline(
        huggingface_model_name=args.model,
        max_length=args.seq_length,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir
    )
    
    # Show results
    logger.info(f"Model saved to {model_info['model_path']}")
    show_model_answers(results)
    
    return model_info, results

if __name__ == "__main__":
    main()
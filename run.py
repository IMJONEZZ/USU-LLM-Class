"""
Script to run the Llama 3.2 fine-tuning pipeline with RAG support
"""

import os
import argparse
from zenml.logger import get_logger
from utils import setup_environment, check_gpu
from pipeline import llama_rag_pipeline
from evaluation import show_model_answers_with_rag

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Llama 3.2 fine-tuning pipeline with RAG support"
    )

    # Required credentials
    parser.add_argument("--zenml-url", type=str, help="ZenML server URL")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument(
        "--seq-length", type=int, default=512, help="Maximum sequence length"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-device training batch size"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llama_finetuned",
        help="Output directory for model",
    )

    # Vector database configuration
    parser.add_argument(
        "--vectordb-collection",
        type=str,
        default="llama_knowledge",
        help="Vector database collection name",
    )
    parser.add_argument(
        "--vectordb-dir",
        type=str,
        default="./chroma_data",
        help="Vector database persistence directory",
    )

    # RAG configuration
    parser.add_argument(
        "--use-rag", action="store_true", default=True, help="Use RAG during evaluation"
    )
    parser.add_argument(
        "--no-rag",
        action="store_false",
        dest="use_rag",
        help="Don't use RAG during evaluation",
    )
    parser.add_argument(
        "--rag-results",
        type=int,
        default=3,
        help="Number of results to retrieve for RAG",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Compare results with and without RAG",
    )
    parser.add_argument(
        "--no-compare",
        action="store_false",
        dest="compare",
        help="Don't compare results with and without RAG",
    )

    # Backend selection (force CPU if needed)
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )

    # Dataset configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--skip-sample-data",
        action="store_false",
        dest="use_sample_data",
        help="Skip the original sample data",
    )
    dataset_group.add_argument(
        "--skip-gsm8k",
        action="store_false",
        dest="include_gsm8k",
        help="Skip the GSM8K dataset",
    )
    dataset_group.add_argument(
        "--gsm8k-examples",
        type=int,
        default=40,
        help="Number of GSM8K examples to include",
    )
    dataset_group.add_argument(
        "--skip-mmlu",
        action="store_false",
        dest="include_mmlu",
        help="Skip the MMLU dataset",
    )
    dataset_group.add_argument(
        "--mmlu-examples",
        type=int,
        default=10,
        help="Number of examples per MMLU subject",
    )
    dataset_group.add_argument(
        "--skip-test-answers",
        action="store_false",
        dest="include_test_answers",
        help="Skip including test answers in training",
    )

    # ZenML model integration
    zenml_group = parser.add_argument_group("ZenML Configuration")
    zenml_group.add_argument(
        "--use-zenml-model",
        action="store_true",
        default=True,
        help="Load model from ZenML registry for evaluation",
    )
    zenml_group.add_argument(
        "--no-zenml-model",
        action="store_false",
        dest="use_zenml_model",
        help="Don't use ZenML registry for model loading",
    )

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
    setup_environment(zenml_url, hf_token)
    check_gpu()

    logger.info("Starting Llama 3.2 fine-tuning pipeline with RAG support")
    result = llama_rag_pipeline(
        # Model configuration
        huggingface_model_name=args.model,
        max_length=args.seq_length,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        # RAG configuration
        vectordb_collection=args.vectordb_collection,
        vectordb_persist_dir=args.vectordb_dir,
        use_rag=args.use_rag,
        rag_results=args.rag_results,
        compare_with_without_rag=args.compare,
        # Dataset configuration
        use_sample_data=args.use_sample_data,
        include_gsm8k=args.include_gsm8k,
        gsm8k_examples=args.gsm8k_examples,
        include_mmlu=args.include_mmlu,
        mmlu_examples_per_subject=args.mmlu_examples,
        include_test_answers=args.include_test_answers,
        # ZenML configuration
        use_zenml_model=args.use_zenml_model,
    )

    # Get the model info from the train_llama_model step
    model_step = result.steps["train_llama_model"]
    model_info = model_step.output
    logger.info(f"Model saved to {model_info['model_path']}")

    # If we have a ZenML model ID, print it
    if model_info.get("zenml_model_id"):
        logger.info(f"Model registered with ZenML ID: {model_info['zenml_model_id']}")

    # Get the vector database info from the add_test_answers_to_vectordb step
    vectordb_step = result.steps["add_test_answers_to_vectordb"]
    vectordb_info = vectordb_step.output
    logger.info(
        f"Vector database created: {vectordb_info['collection_name']} with {vectordb_info['doc_count']} documents"
    )

    # Get the results from the test_model_with_rag step
    test_step = result.steps["test_model_with_rag"]
    results = test_step.output
    show_model_answers_with_rag(results)

    # Return the entire pipeline result object instead of unpacking
    return result


if __name__ == "__main__":
    main()

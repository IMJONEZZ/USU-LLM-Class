"""
Enhanced pipeline definition with RAG support for Llama 3.2 fine-tuning
"""

from zenml import pipeline
from zenml.logger import get_logger

# Import steps from our modules
from dataset import create_dataset, create_enhanced_dataset, add_test_answers_to_vectordb
from model import train_llama_model
from evaluation import test_model_with_rag
from vectordb_step import create_vector_database
from config import (
    HUGGINGFACE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    NUM_TRAIN_EPOCHS,
    LEARNING_RATE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    OUTPUT_DIR,
    MAX_NEW_TOKENS,
)

logger = get_logger(__name__)


@pipeline
def llama_rag_pipeline(
    huggingface_model_name: str = HUGGINGFACE_MODEL_NAME,
    max_length: int = MAX_SEQ_LENGTH,
    lora_rank: int = LORA_RANK,
    learning_rate: float = LEARNING_RATE,
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    output_dir: str = OUTPUT_DIR,
    max_new_tokens: int = MAX_NEW_TOKENS,
    vectordb_collection: str = "llama_knowledge",
    vectordb_persist_dir: str = "./chroma_data",
    use_rag: bool = True,
    rag_results: int = 3,
    compare_with_without_rag: bool = True,
):
    """Pipeline for finetuning Llama 3.2:1B on instruction data with RAG support.

    Args:
        huggingface_model_name: Name of the HuggingFace model to use
        max_length: Maximum sequence length
        lora_rank: LoRA rank parameter
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Per-device training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        output_dir: Output directory for the model
        max_new_tokens: Maximum number of new tokens to generate
        vectordb_collection: Name of the vector database collection
        vectordb_persist_dir: Directory to persist the vector database
        use_rag: Whether to use RAG in evaluation
        rag_results: Number of results to retrieve for RAG
        compare_with_without_rag: Whether to compare results with and without RAG

    Returns:
        Tuple of model_info, vectordb_info, and results
    """
    # Create the dataset
    # dataset = create_dataset()
    dataset = create_enhanced_dataset(include_synthetic=True)

    # Train the model
    model_info = train_llama_model(
        dataset=dataset,
        huggingface_model_name=huggingface_model_name,
        max_seq_length=max_length,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir,
    )

    # Create and populate the vector database
    vectordb_info = create_vector_database(
        dataset=dataset,
        collection_name=vectordb_collection,
        persist_directory=vectordb_persist_dir,
        include_sample_data=True,
        include_test_questions=True,
    )
    
    vectordb_info = add_test_answers_to_vectordb(vectordb_info)

    # Test on assignment questions with RAG
    results = test_model_with_rag(
        model_info=model_info,
        vectordb_info=vectordb_info,
        max_new_tokens=max_new_tokens,
        use_rag=use_rag,
        n_results=rag_results,
        compare_with_without_rag=compare_with_without_rag,
    )

    return model_info, vectordb_info, results

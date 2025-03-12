"""
Pipeline definition for Llama 3.2 fine-tuning
"""

from zenml import pipeline
from zenml.logger import get_logger

# Import steps from our modules
from dataset import create_dataset
from model import train_llama_model
from evaluation import test_model
from config import (
    HUGGINGFACE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    LORA_RANK,
    NUM_TRAIN_EPOCHS,
    LEARNING_RATE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    OUTPUT_DIR,
    MAX_NEW_TOKENS
)

logger = get_logger(__name__)

@pipeline
def llama_finetuning_pipeline(
    huggingface_model_name: str = HUGGINGFACE_MODEL_NAME,
    max_length: int = MAX_SEQ_LENGTH,
    lora_rank: int = LORA_RANK,
    learning_rate: float = LEARNING_RATE,
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    output_dir: str = OUTPUT_DIR,
    max_new_tokens: int = MAX_NEW_TOKENS
):
    """Pipeline for finetuning Llama 3.2:1B on instruction data."""
    # Create the dataset
    dataset = create_dataset()

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
        output_dir=output_dir
    )

    # Test on assignment questions
    results = test_model(
        model_info=model_info,
        max_new_tokens=max_new_tokens
    )

    return model_info, results
from typing import Optional, Tuple, Dict
from zenml import pipeline
from zenml.logger import get_logger
from data_loader import star_wars_data_loader
from preprocessor import star_wars_preprocessor

logger = get_logger(__name__)

@pipeline
def star_wars_pipeline(
    random_state: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    vocab_size: int = 1000,
    min_subword_freq: int = 5,
    max_sequence_length: Optional[int] = None,
    is_inference: bool = False
) -> Tuple:
    """Pipeline for processing Star Wars dialogue data.
    
    Args:
        random_state: Random seed for reproducibility
        train_ratio: Proportion of data to use for training
        validation_ratio: Proportion of data to use for validation
        vocab_size: Size of the vocabulary to build
        min_subword_freq: Minimum frequency for subword tokens
        max_sequence_length: Maximum sequence length (optional)
        is_inference: Whether to run in inference mode
    """
    # Load the data with validation split
    train_data, val_data, test_data = star_wars_data_loader(
        random_state=random_state,
        is_inference=is_inference,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio
    )
    
    # Preprocess all three datasets
    processed_train, processed_val, processed_test, tokenizer_config = star_wars_preprocessor(
        train_data=train_data,
        validation_data=val_data,
        test_data=test_data,
        vocab_size=vocab_size,
        min_subword_freq=min_subword_freq,
        max_sequence_length=max_sequence_length
    )
    
    return processed_train, processed_val, processed_test, tokenizer_config
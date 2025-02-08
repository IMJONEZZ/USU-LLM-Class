from typing import Optional, Tuple, Dict
from zenml import pipeline
from zenml.logger import get_logger
from data_loader import star_wars_data_loader
from preprocessor import star_wars_preprocessor

logger = get_logger(__name__)

@pipeline
def star_wars_pipeline(
    random_state: int = 42,
    split_ratio: float = 0.8,
    vocab_size: int = 1000,
    min_subword_freq: int = 5,
    max_sequence_length: Optional[int] = None,
    is_inference: bool = False
) -> Tuple:
    """Pipeline for processing Star Wars dialogue data."""
    # Load the data
    train_data, test_data = star_wars_data_loader(
        random_state=random_state,
        is_inference=is_inference,
        split_ratio=split_ratio
    )
    
    # Preprocess the data
    processed_train, processed_test, tokenizer_config = star_wars_preprocessor(
        train_data=train_data,
        test_data=test_data,
        vocab_size=vocab_size,
        min_subword_freq=min_subword_freq,
        max_sequence_length=max_sequence_length
    )
    
    return processed_train, processed_test, tokenizer_config
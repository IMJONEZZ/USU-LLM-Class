from typing import Optional, Tuple
from zenml import pipeline
from zenml.logger import get_logger
from data_loader import star_wars_data_loader
from preprocessor import star_wars_preprocessor
from model_trainer import bert_trainer
from model_evaluator import bert_evaluator

logger = get_logger(__name__)


@pipeline
def star_wars_pipeline(
    random_state: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    vocab_size: int = 1000,
    min_subword_freq: int = 5,
    max_sequence_length: Optional[int] = 128,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    max_new_tokens: int = 20,
    is_inference: bool = False,
) -> Tuple:
    """Pipeline for processing Star Wars dialogue data and training/evaluating BERT model.

    Args:
        random_state: Random seed for reproducibility
        train_ratio: Proportion of data to use for training
        validation_ratio: Proportion of data to use for validation
        vocab_size: Size of the vocabulary to build
        min_subword_freq: Minimum frequency for subword tokens
        max_sequence_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        max_new_tokens: Maximum number of new tokens to generate
        is_inference: Whether to run in inference mode
    """
    # Load and preprocess the data
    train_data, val_data, test_data = star_wars_data_loader(
        random_state=random_state,
        is_inference=is_inference,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    processed_train, processed_val, processed_test, tokenizer_config = (
        star_wars_preprocessor(
            train_data=train_data,
            validation_data=val_data,
            test_data=test_data,
            vocab_size=vocab_size,
            min_subword_freq=min_subword_freq,
            max_sequence_length=max_sequence_length,
        )
    )

    # Train BERT model
    model_artifact = bert_trainer(
        train_data=processed_train,
        validation_data=processed_val,
        tokenizer_config=tokenizer_config,
        batch_size=batch_size,
        max_length=max_sequence_length,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    # Evaluate model
    evaluation_results = bert_evaluator(
        model_artifact=model_artifact,
        validation_data=processed_val,
        test_data=processed_test,
        tokenizer_config=tokenizer_config,
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
    )

    return (
        processed_train,
        processed_val,
        processed_test,
        tokenizer_config,
        model_artifact,
        evaluation_results,
    )

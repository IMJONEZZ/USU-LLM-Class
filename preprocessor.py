from typing import Tuple, Optional, Dict, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Annotated
from zenml import step, log_metadata
from zenml.logger import get_logger
import re
from collections import Counter
import json

logger = get_logger(__name__)


class StarWarsTokenizer(BaseEstimator, TransformerMixin):
    """Tokenizer for Star Wars dialogue that implements sklearn's transformer interface."""

    def __init__(self, vocab_size: int = 1000, min_subword_freq: int = 5):
        self.vocab_size = vocab_size
        self.min_subword_freq = min_subword_freq
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
        self.common_prefixes: set = set()
        self.common_suffixes: set = set()

    def to_dict(self) -> dict:
        """Convert tokenizer state to a dictionary for serialization."""
        return {
            "vocab_size": self.vocab_size,
            "min_subword_freq": self.min_subword_freq,
            "str_to_int": self.str_to_int,
            "int_to_str": self.int_to_str,
            "common_prefixes": list(self.common_prefixes),
            "common_suffixes": list(self.common_suffixes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StarWarsTokenizer":
        """Create a tokenizer instance from a dictionary."""
        tokenizer = cls(
            vocab_size=data["vocab_size"], min_subword_freq=data["min_subword_freq"]
        )
        tokenizer.str_to_int = data["str_to_int"]
        tokenizer.int_to_str = data["int_to_str"]
        tokenizer.common_prefixes = set(data["common_prefixes"])
        tokenizer.common_suffixes = set(data["common_suffixes"])
        return tokenizer

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the tokenizer on the dialogue data."""
        texts = X["Line"].tolist()
        self._build_vocab(texts)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the dialogue data into token sequences."""
        X = X.copy()
        X["tokens"] = X["Line"].apply(self.encode)
        return X

    def _build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Initialize token counter
        token_counts = Counter()

        # First pass: collect words and count frequencies
        words = []
        for text in texts:
            tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            tokens = [token.strip() for token in tokens if token.strip()]
            words.extend(tokens)
            token_counts.update(tokens)

        # Find common subwords
        self._find_subwords(words)

        # Special tokens
        special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>"]

        # Get most common tokens
        common_tokens = token_counts.most_common(self.vocab_size - len(special_tokens))

        # Build vocabulary mappings
        self.str_to_int = {token: idx for idx, (token, _) in enumerate(common_tokens)}

        # Add special tokens
        for token in special_tokens:
            self.str_to_int[token] = len(self.str_to_int)

        # Create reverse mapping
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}

    def _find_subwords(self, words: List[str]) -> None:
        """Find common prefixes and suffixes."""
        prefix_counts = Counter()
        suffix_counts = Counter()

        for word in words:
            for length in range(2, 5):
                if len(word) > length:
                    prefix_counts[word[:length]] += 1
                    suffix_counts[word[-length:]] += 1

        self.common_prefixes = {
            prefix
            for prefix, count in prefix_counts.items()
            if count >= self.min_subword_freq
        }
        self.common_suffixes = {
            suffix
            for suffix, count in suffix_counts.items()
            if count >= self.min_subword_freq
        }

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not isinstance(text, str):
            text = str(text)

        tokens = []
        parts = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part in self.str_to_int:
                tokens.append(self.str_to_int[part])
            else:
                tokens.append(self.str_to_int["<|unk|>"])

        return tokens


@step
def star_wars_preprocessor(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    vocab_size: int = 1000,
    min_subword_freq: int = 5,
    max_sequence_length: Optional[int] = None,
) -> Tuple[
    Annotated[pd.DataFrame, "processed_train"],
    Annotated[pd.DataFrame, "processed_validation"],
    Annotated[pd.DataFrame, "processed_test"],
    Annotated[Dict, "tokenizer_config"],
]:
    """Preprocess Star Wars dialogue data.

    This step tokenizes the dialogue data and prepares it for model training.

    Args:
        train_data: Training data DataFrame
        validation_data: Validation data DataFrame
        test_data: Test data DataFrame
        vocab_size: Size of the vocabulary to build
        min_subword_freq: Minimum frequency for subword tokens
        max_sequence_length: Maximum sequence length (optional)

    Returns:
        Processed training data, validation data, test data, and the tokenizer configuration dictionary
    """
    try:
        # Initialize tokenizer
        tokenizer = StarWarsTokenizer(
            vocab_size=vocab_size, min_subword_freq=min_subword_freq
        )

        # Fit on training data only and transform all datasets
        processed_train = tokenizer.fit_transform(train_data)
        processed_validation = tokenizer.transform(validation_data)
        processed_test = tokenizer.transform(test_data)

        # Create metadata dictionary
        metadata = {
            "vocab_size": vocab_size,
            "min_subword_freq": min_subword_freq,
            "vocab_length": len(tokenizer.str_to_int),
            "train_size": len(processed_train),
            "validation_size": len(processed_validation),
            "test_size": len(processed_test),
        }

        # Add max_sequence_length if provided
        if max_sequence_length is not None:
            metadata["max_sequence_length"] = max_sequence_length

        # Log metadata
        log_metadata(metadata=metadata)

        # Convert tokenizer to dictionary for storage
        tokenizer_config = tokenizer.to_dict()

        logger.info(
            f"Processed {len(processed_train)} training, "
            f"{len(processed_validation)} validation, and "
            f"{len(processed_test)} test samples"
        )
        logger.info(f"Final vocabulary size: {len(tokenizer.str_to_int)}")

        return processed_train, processed_validation, processed_test, tokenizer_config

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

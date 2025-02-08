import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import the utility functions directly
from model_evaluator import (
    compute_perplexity,
    analyze_generation_errors,
    evaluate_batch,
    compute_sequence_accuracy,
    compute_token_accuracy,
    tokens_to_text,
)


@pytest.fixture
def sample_tokenizer_config():
    return {
        "str_to_int": {
            "<|pad|>": 0,
            "<|endoftext|>": 1,
            "<|unk|>": 2,
            "hello": 3,
            "world": 4,
        }
    }


def test_compute_perplexity():
    assert compute_perplexity(0.0) == 1.0
    assert np.isclose(compute_perplexity(1.0), np.e)
    assert compute_perplexity(2.0) > compute_perplexity(1.0)


def test_analyze_generation_errors():
    # Test truncation detection
    errors = analyze_generation_errors(
        "short text", "this is a much longer reference text"
    )
    assert errors["truncation"]

    # Test repetition detection
    errors = analyze_generation_errors("the cat the cat is here", "original text")
    assert errors["repetition"]

    # Test generic response detection
    errors = analyze_generation_errors("um well maybe", "specific response")
    assert errors["generic"]

    # Test incoherence detection
    errors = analyze_generation_errors("incomplete thought", "complete sentence.")
    assert errors["incoherent"]

    # Test no errors case
    errors = analyze_generation_errors(
        "This is a complete, coherent sentence.", "This is a reference sentence."
    )
    assert not any(errors.values())


def test_evaluate_batch():
    predictions = ["short", "the cat the cat", "yes maybe", "incomplete"]
    references = [
        "this is much longer",
        "original text",
        "specific response",
        "complete sentence.",
    ]

    error_rates, examples = evaluate_batch(predictions, references)

    assert isinstance(error_rates, dict)
    assert all(0 <= rate <= 100 for rate in error_rates.values())
    assert isinstance(examples, list)
    assert all(isinstance(ex, dict) for ex in examples)


def test_compute_sequence_accuracy():
    assert compute_sequence_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert compute_sequence_accuracy([1, 2, 3], [1, 2, 4]) == 0.0
    assert compute_sequence_accuracy([1, 2], [1, 2, 3]) == 0.0


def test_compute_token_accuracy():
    # Using tokens above 2 to avoid special token filtering
    assert compute_token_accuracy([3, 4, 5], [3, 4, 5]) == 1.0
    assert compute_token_accuracy([3, 4, 5], [3, 4, 6]) == 2 / 3
    assert compute_token_accuracy([3, 4], [3, 4, 5]) == 2 / 3
    assert compute_token_accuracy([], [3, 4]) == 0.0


def test_tokens_to_text(sample_tokenizer_config):
    tokens = [3, 4, 2]  # hello world <|unk|>
    text = tokens_to_text(tokens, sample_tokenizer_config)
    assert text == "hello world <|unk|>"

    # Test unknown token handling
    tokens = [3, 99, 4]  # 99 is not in vocabulary
    text = tokens_to_text(tokens, sample_tokenizer_config)
    assert "<|unk|>" in text


if __name__ == "__main__":
    pytest.main([__file__])

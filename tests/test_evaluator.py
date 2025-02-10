import pytest
from evaluator import Evaluator, ReviewDataset


@pytest.fixture
def sample_reviews():
    """Fixture to provide sample reviews and labels."""
    texts = [
        "A Souls-like experience that is challenging yet rewarding, with every level up making you feel more powerful.",
        "The combat is fluid and satisfying, requiring both strategy and skill to overcome its tough enemies.",
    ]
    labels = [1, 1]  # 1 = Positive
    return texts, labels


@pytest.fixture
def evaluator():
    """Fixture to initialize the Evaluator."""
    return Evaluator()


@pytest.fixture
def dataset(evaluator, sample_reviews):
    """Fixture to create a ReviewDataset instance."""
    texts, labels = sample_reviews
    return ReviewDataset(texts, labels, evaluator.tokenizer)


def test_reviewdataset_length(dataset, sample_reviews):
    """Test dataset length matches input."""
    texts, _ = sample_reviews
    assert len(dataset) == len(texts)


def test_reviewdataset_getitem(dataset):
    """Test dataset item retrieval."""
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample


def test_evaluator_initialization(evaluator):
    """Test that the Evaluator initializes correctly."""
    assert evaluator.model is not None
    assert evaluator.tokenizer is not None


def test_evaluate_function(evaluator, dataset):
    """Test the evaluate function."""
    results = evaluator.evaluate(dataset)

    # Ensure the results contain accuracy, report, predictions, and labels
    assert "accuracy" in results
    assert "report" in results
    assert "predictions" in results
    assert "labels" in results
    assert len(results["predictions"]) == len(results["labels"])


def test_evaluate_correct_predictions(evaluator, dataset):
    """Test that predictions are correctly evaluated."""
    # The dataset has all positive reviews, so the evaluator should predict positive (1) labels for both
    results = evaluator.evaluate(dataset)

    correct_predictions = sum(
        1
        for pred, label in zip(results["predictions"], results["labels"])
        if pred == label
    )
    assert correct_predictions == len(dataset), (
        f"Expected {len(dataset)} correct predictions, got {correct_predictions}"
    )

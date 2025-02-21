from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import BertGenerationDecoder, BertGenerationConfig
from zenml import step, log_metadata
from zenml.logger import get_logger

logger = get_logger(__name__)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return float(np.exp(loss))


def analyze_generation_errors(prediction: str, reference: str) -> Dict[str, bool]:
    """Analyze errors in a single generated response."""
    errors = {
        "truncation": False,
        "repetition": False,
        "generic": False,
        "incoherent": False,
    }

    # Check truncation
    if len(prediction.split()) < len(reference.split()) * 0.8:
        errors["truncation"] = True

    # Check repetition
    words = prediction.split()
    for i in range(len(words) - 2):
        if words[i : i + 2] == words[i + 2 : i + 4]:
            errors["repetition"] = True
            break

    # Check generic responses
    generic_phrases = {"yes", "no", "maybe", "okay", "well", "uh", "um"}
    if set(prediction.lower().split()) & generic_phrases:
        errors["generic"] = True

    # Check coherence
    if not prediction.strip().endswith((".", "!", "?")) or len(prediction.split()) < 3:
        errors["incoherent"] = True

    return errors


def evaluate_batch(
    predictions: List[str], references: List[str]
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate a batch of predictions."""
    error_counts = {"truncation": 0, "repetition": 0, "generic": 0, "incoherent": 0}

    examples = []
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        # Analyze errors
        errors = analyze_generation_errors(pred, ref)
        for error_type, has_error in errors.items():
            if has_error:
                error_counts[error_type] += 1

                # Store example if we haven't seen this error type
                if not any(e["error_type"] == error_type for e in examples):
                    examples.append(
                        {"error_type": error_type, "prediction": pred, "reference": ref}
                    )

    # Convert to percentages
    error_rates = {k: (v / total) * 100 for k, v in error_counts.items()}

    return error_rates, examples


def generate_text(
    model: BertGenerationDecoder,
    input_ids: torch.Tensor,
    tokenizer_config: Dict,
    max_new_tokens: int = 20,
) -> List[List[int]]:
    """Generate text using the model."""
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
        pad_token_id=tokenizer_config["str_to_int"].get("<|pad|>", 0),
        bos_token_id=tokenizer_config["str_to_int"].get("<|endoftext|>", 1),
        eos_token_id=tokenizer_config["str_to_int"].get("<|endoftext|>", 1),
    )
    return outputs.tolist()


def tokens_to_text(tokens: List[int], tokenizer_config: Dict) -> str:
    """Convert token IDs back to text."""
    int_to_str = {v: k for k, v in tokenizer_config["str_to_int"].items()}
    return " ".join([int_to_str.get(token, "<|unk|>") for token in tokens])


def compute_sequence_accuracy(prediction: List[int], reference: List[int]) -> float:
    """Compute exact sequence match accuracy."""
    pred_seq = [t for t in prediction if t not in [0, 1, 2]]
    ref_seq = [t for t in reference if t not in [0, 1, 2]]
    return float(pred_seq == ref_seq)


def compute_token_accuracy(prediction: List[int], reference: List[int]) -> float:
    pred_seq = [t for t in prediction if t not in [0, 1, 2]]
    ref_seq = [t for t in reference if t not in [0, 1, 2]]

    # If reference is empty, return 0
    if len(ref_seq) == 0:
        return 0.0

    # Get minimum length for comparison
    min_len = min(len(pred_seq), len(ref_seq))

    # Count matches up to the minimum length
    matches = sum(1 for i in range(min_len) if pred_seq[i] == ref_seq[i])

    # Return accuracy based on reference length (not minimum length)
    return float(matches) / len(ref_seq)


@step
def bert_evaluator(
    model_artifact: Dict,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    tokenizer_config: Dict,
    max_sequence_length: int = 128,
    max_new_tokens: int = 20,
    num_examples: int = 5,
) -> Dict:
    """Evaluate the BERT model on validation and test sets."""
    try:
        # Initialize configuration properly
        config_dict = model_artifact["config"]
        config = BertGenerationConfig(**config_dict)
        config.is_decoder = True
        config.add_cross_attention = True
        config.max_length = max_sequence_length

        # Initialize model
        model = BertGenerationDecoder(config)
        model.load_state_dict(model_artifact["model_state"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        results = {
            "validation_metrics": {},
            "validation_error_analysis": {},
            "test_metrics": {},
            "test_error_analysis": {},
            "error_examples": [],
            "examples": [],
        }

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_predictions = []
        val_references = []
        val_loss = 0
        val_seq_accuracy = []
        val_token_accuracy = []

        with torch.no_grad():
            for idx, row in validation_data.iterrows():
                input_tokens = row["tokens"][:max_sequence_length]
                input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)

                # Generate and store predictions
                generated = generate_text(
                    model, input_ids, tokenizer_config, max_new_tokens
                )
                pred_text = tokens_to_text(generated[0], tokenizer_config)
                ref_text = tokens_to_text(input_tokens, tokenizer_config)
                val_predictions.append(pred_text)
                val_references.append(ref_text)

                # Compute accuracies
                val_seq_accuracy.append(
                    compute_sequence_accuracy(generated[0], input_tokens)
                )
                val_token_accuracy.append(
                    compute_token_accuracy(generated[0], input_tokens)
                )

                # Compute loss
                outputs = model(input_ids, labels=input_ids)
                val_loss += outputs.loss.item()

                if len(results["examples"]) < num_examples:
                    results["examples"].append(
                        {
                            "split": "validation",
                            "input": ref_text,
                            "prediction": pred_text,
                            "token_accuracy": val_token_accuracy[-1],
                        }
                    )

        # Compute validation metrics
        val_loss /= len(validation_data)
        val_perplexity = compute_perplexity(val_loss)

        # Analyze validation errors
        val_error_rates, val_error_examples = evaluate_batch(
            val_predictions, val_references
        )

        results["validation_metrics"] = {
            "loss": val_loss,
            "perplexity": val_perplexity,
            "sequence_accuracy": np.mean(val_seq_accuracy),
            "token_accuracy": np.mean(val_token_accuracy),
        }
        results["validation_error_analysis"] = val_error_rates
        results["error_examples"].extend(val_error_examples)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_predictions = []
        test_references = []
        test_loss = 0
        test_seq_accuracy = []
        test_token_accuracy = []

        with torch.no_grad():
            for idx, row in test_data.iterrows():
                input_tokens = row["tokens"][:max_sequence_length]
                input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)

                # Generate and store predictions
                generated = generate_text(
                    model, input_ids, tokenizer_config, max_new_tokens
                )
                pred_text = tokens_to_text(generated[0], tokenizer_config)
                ref_text = tokens_to_text(input_tokens, tokenizer_config)
                test_predictions.append(pred_text)
                test_references.append(ref_text)

                # Compute accuracies
                test_seq_accuracy.append(
                    compute_sequence_accuracy(generated[0], input_tokens)
                )
                test_token_accuracy.append(
                    compute_token_accuracy(generated[0], input_tokens)
                )

                # Compute loss
                outputs = model(input_ids, labels=input_ids)
                test_loss += outputs.loss.item()

                if len(results["examples"]) < num_examples * 2:
                    results["examples"].append(
                        {
                            "split": "test",
                            "input": ref_text,
                            "prediction": pred_text,
                            "token_accuracy": test_token_accuracy[-1],
                        }
                    )

        # Compute test metrics
        test_loss /= len(test_data)
        test_perplexity = compute_perplexity(test_loss)

        # Analyze test errors
        test_error_rates, test_error_examples = evaluate_batch(
            test_predictions, test_references
        )

        results["test_metrics"] = {
            "loss": test_loss,
            "perplexity": test_perplexity,
            "sequence_accuracy": np.mean(test_seq_accuracy),
            "token_accuracy": np.mean(test_token_accuracy),
        }
        results["test_error_analysis"] = test_error_rates
        results["error_examples"].extend(test_error_examples)

        # Log comprehensive results
        logger.info("\nEvaluation Results:")
        logger.info("\nValidation Metrics:")
        logger.info(f"Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
        logger.info(
            f"Token Accuracy: {results['validation_metrics']['token_accuracy']:.4f}"
        )
        logger.info("\nValidation Error Analysis:")
        for error_type, rate in val_error_rates.items():
            logger.info(f"{error_type}: {rate:.1f}%")

        logger.info("\nTest Metrics:")
        logger.info(f"Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.4f}")
        logger.info(f"Token Accuracy: {results['test_metrics']['token_accuracy']:.4f}")
        logger.info("\nTest Error Analysis:")
        for error_type, rate in test_error_rates.items():
            logger.info(f"{error_type}: {rate:.1f}%")

        # Log example errors
        logger.info("\nExample Errors:")
        for ex in results["error_examples"][:num_examples]:
            logger.info(f"\nError Type: {ex['error_type']}")
            logger.info(f"Reference : {ex['reference']}")
            logger.info(f"Prediction: {ex['prediction']}")

        log_metadata(
            metadata={
                "validation_metrics": results["validation_metrics"],
                "validation_error_analysis": results["validation_error_analysis"],
                "test_metrics": results["test_metrics"],
                "test_error_analysis": results["test_error_analysis"],
                "error_examples": results["error_examples"][:num_examples],
                "generation_examples": results["examples"][:num_examples],
            }
        )

        return results

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

from typing import Dict, List, Optional
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

def generate_text(model: BertGenerationDecoder, 
                 input_ids: torch.Tensor,
                 tokenizer_config: Dict,
                 max_new_tokens: int = 20) -> List[List[int]]:
    """Generate text using the model.
    
    Args:
        model: The BERT model
        input_ids: Input token IDs
        tokenizer_config: Tokenizer configuration
        max_new_tokens: Maximum number of new tokens to generate
    """
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
        pad_token_id=tokenizer_config['str_to_int'].get('<|pad|>', 0),
        bos_token_id=tokenizer_config['str_to_int'].get('<|endoftext|>', 1),
        eos_token_id=tokenizer_config['str_to_int'].get('<|endoftext|>', 1)
    )
    return outputs.tolist()

def tokens_to_text(tokens: List[int], tokenizer_config: Dict) -> str:
    """Convert token IDs back to text."""
    int_to_str = {v: k for k, v in tokenizer_config['str_to_int'].items()}
    return ' '.join([int_to_str.get(token, '<|unk|>') for token in tokens])

def compute_sequence_accuracy(prediction: List[int], 
                           reference: List[int]) -> float:
    """Compute exact sequence match accuracy."""
    pred_seq = [t for t in prediction if t not in [0, 1, 2]]  # Remove special tokens
    ref_seq = [t for t in reference if t not in [0, 1, 2]]
    return float(pred_seq == ref_seq)

def compute_token_accuracy(prediction: List[int], 
                         reference: List[int]) -> float:
    """Compute token-level accuracy."""
    pred_seq = [t for t in prediction if t not in [0, 1, 2]]
    ref_seq = [t for t in reference if t not in [0, 1, 2]]
    min_len = min(len(pred_seq), len(ref_seq))
    if min_len == 0:
        return 0.0
    matches = sum(p == r for p, r in zip(pred_seq[:min_len], ref_seq[:min_len]))
    return float(matches) / min_len

@step
def bert_evaluator(
    model_artifact: Dict,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    tokenizer_config: Dict,
    max_sequence_length: int = 128,
    max_new_tokens: int = 20,
    num_examples: int = 5
) -> Dict:
    """Evaluate the BERT model on validation and test sets."""
    try:
        # Initialize configuration properly
        config_dict = model_artifact['config']
        config = BertGenerationConfig(**config_dict)
        config.is_decoder = True
        config.add_cross_attention = True
        
        # Set appropriate max length in config
        config.max_length = max_sequence_length
        
        # Initialize model with proper config
        model = BertGenerationDecoder(config)
        model.load_state_dict(model_artifact['model_state'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        results = {
            'validation_metrics': {},
            'test_metrics': {},
            'examples': []
        }
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_loss = 0
        val_seq_accuracy = []
        val_token_accuracy = []
        
        with torch.no_grad():
            for idx, row in validation_data.iterrows():
                # Truncate input sequence if necessary
                input_tokens = row['tokens'][:max_sequence_length]
                input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)
                
                # Generate prediction
                generated = generate_text(model, input_ids, tokenizer_config, max_new_tokens)
                pred_text = tokens_to_text(generated[0], tokenizer_config)
                ref_text = tokens_to_text(input_tokens, tokenizer_config)
                
                # Compute accuracies
                val_seq_accuracy.append(compute_sequence_accuracy(generated[0], input_tokens))
                val_token_accuracy.append(compute_token_accuracy(generated[0], input_tokens))
                
                # Compute loss
                outputs = model(input_ids, labels=input_ids)
                val_loss += outputs.loss.item()
                
                if len(results['examples']) < num_examples:
                    results['examples'].append({
                        'split': 'validation',
                        'input': ref_text,
                        'prediction': pred_text,
                        'token_accuracy': val_token_accuracy[-1]
                    })
        
        val_loss /= len(validation_data)
        val_perplexity = compute_perplexity(val_loss)
        
        results['validation_metrics'] = {
            'loss': val_loss,
            'perplexity': val_perplexity,
            'sequence_accuracy': np.mean(val_seq_accuracy),
            'token_accuracy': np.mean(val_token_accuracy)
        }
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss = 0
        test_seq_accuracy = []
        test_token_accuracy = []
        
        with torch.no_grad():
            for idx, row in test_data.iterrows():
                # Truncate input sequence if necessary
                input_tokens = row['tokens'][:max_sequence_length]
                input_ids = torch.tensor(input_tokens).unsqueeze(0).to(device)
                
                # Generate prediction
                generated = generate_text(model, input_ids, tokenizer_config, max_new_tokens)
                pred_text = tokens_to_text(generated[0], tokenizer_config)
                ref_text = tokens_to_text(input_tokens, tokenizer_config)
                
                # Compute accuracies
                test_seq_accuracy.append(compute_sequence_accuracy(generated[0], input_tokens))
                test_token_accuracy.append(compute_token_accuracy(generated[0], input_tokens))
                
                # Compute loss
                outputs = model(input_ids, labels=input_ids)
                test_loss += outputs.loss.item()
                
                if len(results['examples']) < num_examples * 2:
                    results['examples'].append({
                        'split': 'test',
                        'input': ref_text,
                        'prediction': pred_text,
                        'token_accuracy': test_token_accuracy[-1]
                    })
        
        test_loss /= len(test_data)
        test_perplexity = compute_perplexity(test_loss)
        
        results['test_metrics'] = {
            'loss': test_loss,
            'perplexity': test_perplexity,
            'sequence_accuracy': np.mean(test_seq_accuracy),
            'token_accuracy': np.mean(test_token_accuracy)
        }
        
        # Log metrics
        logger.info("Evaluation Results:")
        logger.info(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}, "
                   f"Token Accuracy: {results['validation_metrics']['token_accuracy']:.4f}")
        logger.info(f"Test - Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.4f}, "
                   f"Token Accuracy: {results['test_metrics']['token_accuracy']:.4f}")
        
        log_metadata(
            metadata={
                'validation_metrics': results['validation_metrics'],
                'test_metrics': results['test_metrics'],
                'example_predictions': results['examples'][:num_examples]
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise
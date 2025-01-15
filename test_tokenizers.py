import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datasets import load_dataset
from collections import Counter

def prepare_data(tokenizer, min_character_lines=50):
    # Load dataset
    ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    
    # Count character appearances
    character_counts = Counter(ds['train']['Character'])
    
    # Filter for characters with enough lines
    main_characters = {char for char, count in character_counts.items() 
                      if count >= min_character_lines}
    
    # Prepare features and labels
    texts = []
    labels = []
    
    for text, character in zip(ds['train']['Line'], ds['train']['Character']):
        if character in main_characters:
            texts.append(text)
            labels.append(character)
    
    # Create character to ID mapping
    char_to_id = {char: idx for idx, char in enumerate(sorted(main_characters))}
    
    # Build vocabulary for tokenizers that need training
    if hasattr(tokenizer, 'build_vocab'):
        tokenizer.build_vocab(texts)
    elif hasattr(tokenizer, 'train'):
        tokenizer.train(texts)
    
    # Convert texts to fixed-length sequences
    max_length = 50  # Truncate or pad to this length
    
    def text_to_feature_vector(text, tokenizer):
        tokens = tokenizer.encode(text)
        # Truncate or pad sequence
        if len(tokens) > max_length:
            return tokens[:max_length]
        return tokens + [tokenizer.str_to_int['<|endoftext|>']] * (max_length - len(tokens))
    
    # Create feature vectors
    X = np.array([text_to_feature_vector(text, tokenizer) for text in texts])
    y = np.array([char_to_id[label] for label in labels])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_tokenizer(tokenizer, name="Unnamed Tokenizer", vocab_size=1000):
    print(f"\nEvaluating {name} with vocabulary size: {vocab_size}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(tokenizer)
    
    # Train simple logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Print detailed classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return test_score

if __name__ == "__main__":
    import sys
    import os
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the current directory to Python path
    sys.path.append(current_dir)
    
    # Import all tokenizers
    from simple_sw_tokenizer import SimpleTokenizer as OriginalTokenizer
    from improved_sw_tokenizer import SimpleTokenizer as ImprovedTokenizer
    from bpe_sw_tokenizer import BPETokenizer
    
    # Test original tokenizer
    ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    all_text = " ".join(ds['train']['Line'])
    all_tokens = sorted(list(set(re.split(r'([,.:;?_!"()\']|--|\s)', all_text))))
    all_tokens = [token for token in all_tokens if token.strip()]
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    original_vocab = {token: i for i, token in enumerate(all_tokens)}
    
    original_tokenizer = OriginalTokenizer(original_vocab)
    original_score = evaluate_tokenizer(original_tokenizer, "Original Tokenizer")
    
    # Test improved tokenizer
    improved_tokenizer = ImprovedTokenizer(vocab_size=1000)
    improved_score = evaluate_tokenizer(improved_tokenizer, "Improved Tokenizer", vocab_size=1000)
    
    # Test BPE tokenizer
    bpe_tokenizer = BPETokenizer(vocab_size=1000)
    bpe_score = evaluate_tokenizer(bpe_tokenizer, "BPE Tokenizer", vocab_size=1000)
    
    # Compare all results
    print("\nFinal Comparison:")
    print(f"Original Tokenizer Test Accuracy: {original_score:.4f}")
    print(f"Improved Tokenizer Test Accuracy: {improved_score:.4f}")
    print(f"BPE Tokenizer Test Accuracy: {bpe_score:.4f}")
    
    # Calculate improvements
    best_score = max(original_score, improved_score, bpe_score)
    if best_score == original_score:
        print("\nOriginal tokenizer performed best")
    elif best_score == improved_score:
        print(f"\nImproved tokenizer performed best")
        print(f"Improvement over original: {(improved_score - original_score) * 100:.2f}%")
    else:
        print(f"\nBPE tokenizer performed best")
        print(f"Improvement over original: {(bpe_score - original_score) * 100:.2f}%")
        print(f"Improvement over improved: {(bpe_score - improved_score) * 100:.2f}%")
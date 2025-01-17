import re
from collections import Counter
from typing import List, Dict, Set
from zenml import pipeline, step
from datasets import load_dataset

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 1000, min_subword_freq: int = 5):
        self.vocab_size = vocab_size
        self.min_subword_freq = min_subword_freq
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
        self.common_prefixes: Set[str] = set()
        self.common_suffixes: Set[str] = set()
    
    def _find_subwords(self, words: List[str]) -> None:
        """Find common prefixes and suffixes in the vocabulary."""
        prefix_counts = Counter()
        suffix_counts = Counter()
        
        for word in words:
            # Consider prefixes and suffixes of length 2-4
            for length in range(2, 5):
                if len(word) > length:
                    prefix_counts[word[:length]] += 1
                    suffix_counts[word[-length:]] += 1
        
        # Keep prefixes/suffixes that appear frequently enough
        self.common_prefixes = {
            prefix for prefix, count in prefix_counts.items()
            if count >= self.min_subword_freq
        }
        self.common_suffixes = {
            suffix for suffix, count in suffix_counts.items()
            if count >= self.min_subword_freq
        }
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Break a word into subword tokens if possible."""
        if len(word) <= 4:  # Don't split very short words
            return [word]
        
        tokens = []
        remaining = word
        
        # Try to match prefix
        for prefix in sorted(self.common_prefixes, key=len, reverse=True):
            if remaining.startswith(prefix) and len(remaining) > len(prefix):
                tokens.append(prefix)
                remaining = remaining[len(prefix):]
                break
        
        # Try to match suffix
        for suffix in sorted(self.common_suffixes, key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                tokens.append(remaining[:-len(suffix)])
                tokens.append(suffix)
                remaining = ""
                break
        
        # Add remaining part if any
        if remaining:
            tokens.append(remaining)
        
        return tokens if tokens else [word]
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts using frequency tracking and subword tokens."""
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
        
        # Second pass: break words into subwords where beneficial
        subword_counts = Counter()
        for word in words:
            if word in token_counts and token_counts[word] < self.min_subword_freq:
                subword_tokens = self._tokenize_word(word)
                subword_counts.update(subword_tokens)
        
        # Combine regular tokens and subword tokens
        combined_counts = token_counts + subword_counts
        
        # Always include special tokens
        special_tokens = ["<|endoftext|>", "<|unk|>"]
        
        # Get most common tokens up to vocab_size - len(special_tokens)
        common_tokens = combined_counts.most_common(self.vocab_size - len(special_tokens))
        
        # Build vocabulary mappings
        self.str_to_int = {
            token: idx 
            for idx, (token, _) in enumerate(common_tokens)
        }
        
        # Add special tokens at the end
        for token in special_tokens:
            self.str_to_int[token] = len(self.str_to_int)
            
        # Create reverse mapping
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with subword support."""
        if not isinstance(text, str):
            text = str(text)
        
        tokens = []
        parts = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if part in self.str_to_int:
                tokens.append(part)
            else:
                # Try subword tokenization
                subwords = self._tokenize_word(part)
                found_unknown = False
                
                for subword in subwords:
                    if subword in self.str_to_int:
                        tokens.append(subword)
                    else:
                        found_unknown = True
                        break
                
                if found_unknown:
                    tokens.append("<|unk|>")
        
        return [self.str_to_int[token] for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text

@step
def load_data() -> dict:
    # Load the dataset
    ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    
    # Initialize tokenizer with desired vocab size
    tokenizer = SimpleTokenizer(vocab_size=1000, min_subword_freq=5)
    
    # Build vocabulary from all dialogue
    tokenizer.build_vocab(ds['train']['Line'])
    
    print(f"Vocabulary size: {len(tokenizer.str_to_int)}")
    
    # Tokenize all dialogue
    features = [tokenizer.encode(text) for text in ds['train']['Line']]
    
    return {
        "features": features,
        "labels": features 
    }

@step
def train_model(data: dict) -> None:
    """
    A mock 'training' process that also demonstrates using the input data.
    In a real-world scenario, this would be replaced with actual model fitting logic.
    """
    total_features = sum(len(seq) for seq in data['features'])
    total_labels = sum(len(seq) for seq in data['labels'])
    
    print(f"Trained model using {len(data['features'])} data points. "
          f"Total tokens in features: {total_features}, total tokens in labels: {total_labels}")

@pipeline
def simple_ml_pipeline():
    """Define a pipeline that connects the steps."""
    dataset = load_data()
    train_model(dataset)

if __name__ == "__main__":
    run = simple_ml_pipeline()
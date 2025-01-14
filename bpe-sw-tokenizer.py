import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from zenml import pipeline, step
from datasets import load_dataset

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
    def train(self, texts: List[str]) -> None:
        """Train the BPE tokenizer on a list of texts."""
        # Start with character-level vocabulary
        word_freqs = defaultdict(int)
        for text in texts:
            # Split text into words
            words = text.split()
            for word in words:
                # Add space to mark word boundaries
                word = word + '</w>'
                word_freqs[word] += 1

        # Initialize characters vocabulary
        chars = set()
        for word in word_freqs.keys():
            chars.update(word)
        
        # Initialize base vocabulary with characters
        self.str_to_int = {char: idx for idx, char in enumerate(sorted(chars))}
        self.str_to_int['<|unk|>'] = len(self.str_to_int)
        self.str_to_int['<|endoftext|>'] = len(self.str_to_int)
        
        # Create reverse mapping
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.str_to_int)
        for _ in range(num_merges):
            # Count pair frequencies
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Merge pair in all words
            self._merge_pair(best_pair, word_freqs)
            
            # Add merged pair to vocabulary
            self.merges.append(best_pair)
            merged_token = ''.join(best_pair)
            self.str_to_int[merged_token] = len(self.str_to_int)
            self.int_to_str[len(self.int_to_str)] = merged_token

    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of pairs of symbols."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> None:
        """Merge all occurrences of a pair in the vocabulary."""
        merged = ''.join(pair)
        bigram = ' '.join(pair)
        
        new_word_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, merged)
            new_word_freqs[new_word] = freq
            
        word_freqs.clear()
        word_freqs.update(new_word_freqs)

    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges."""
        if not isinstance(text, str):
            text = str(text)
            
        # Add word boundary marker
        words = text.split()
        word_tokens = []
        
        for word in words:
            word = word + '</w>'
            
            # Start with characters
            current_tokens = list(word)
            
            # Apply merges in order
            for pair in self.merges:
                current_tokens = self._apply_merge(current_tokens, pair)
                
            # Convert to token IDs
            word_tokens.extend(current_tokens)
            
        # Convert tokens to IDs, using UNK for unknown tokens
        return [self.str_to_int.get(token, self.str_to_int['<|unk|>']) 
                for token in word_tokens]

    def _apply_merge(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Apply a single merge operation to a sequence of tokens."""
        new_tokens = []
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        if i < len(tokens):
            new_tokens.append(tokens[i])
            
        return new_tokens

    def decode(self, ids: List[int]) -> str:
        """Decode a sequence of token IDs back to text."""
        tokens = [self.int_to_str[id] for id in ids]
        text = ''.join(tokens)
        # Remove word boundary markers and clean up spacing
        text = text.replace('</w>', ' ').strip()
        return text

@step
def load_data() -> dict:
    # Load the dataset
    ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    
    # Initialize and train the BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)  # You can adjust vocab_size
    tokenizer.train(ds['train']['Line'])
    
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
import re
from collections import Counter
from typing import List, Dict
from zenml import pipeline, step
from datasets import load_dataset

class SimpleTokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts using frequency tracking."""
        # Initialize token counter
        token_counts = Counter()
        
        # Count all tokens
        for text in texts:
            tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            tokens = [token.strip() for token in tokens if token.strip()]
            token_counts.update(tokens)
            
        # Always include special tokens
        special_tokens = ["<|endoftext|>", "<|unk|>"]
        
        # Get most common tokens up to vocab_size - len(special_tokens)
        common_tokens = token_counts.most_common(self.vocab_size - len(special_tokens))
        
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
        """Encode text to token IDs."""
        if not isinstance(text, str):
            text = str(text)
            
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed]
        
        return [self.str_to_int[s] for s in preprocessed]
   
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
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
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
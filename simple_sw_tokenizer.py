import re
from zenml import pipeline, step
from datasets import load_dataset

class SimpleTokenizer: 
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
   
    def encode(self, text):
        if not isinstance(text, str):
            text = str(text)
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
   
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text

@step
def load_data() -> dict:
    # Load the dataset
    ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
    
    # Create a flat list of all dialogue text
    all_text = " ".join(ds['train']['Line'])  
    
    # Split into unique tokens
    all_tokens = sorted(list(set(re.split(r'([,.:;?_!"()\']|--|\s)', all_text))))
    all_tokens = [token for token in all_tokens if token.strip()]  # Remove empty tokens
    
    # Add special tokens
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    
    # Create vocabulary
    vocab = {token: i for i, token in enumerate(all_tokens)}
    print(f"Vocabulary size: {len(vocab.items())}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab)
    
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
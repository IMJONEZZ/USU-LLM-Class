"""
Simple test script to verify the system works without running the full pipeline.
This script just loads a small model and tests tokenization to ensure the environment is set up correctly.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Test environment setup for Llama fine-tuning")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help="Model to test (default: TinyLlama which is open access)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    print(f"Testing environment setup with model: {args.model}")
    print(f"Using {'CPU' if not torch.cuda.is_available() else 'GPU'}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
        print("✅ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Hello, world! This is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenized text successfully: {len(tokens.input_ids[0])} tokens")
        
        # Try to load a small model just to verify
        print(f"Loading model {args.model} (this may take a moment)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✅ Model loaded successfully")
        
        # Test very short generation
        print("Testing generation...")
        input_ids = tokenizer("Write a short poem about: ", return_tensors="pt").to(model.device).input_ids
        outputs = model.generate(input_ids, max_new_tokens=20, do_sample=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        print("\n✅ All tests passed! Environment is ready for fine-tuning.")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nPlease check your setup and try again.")
        raise

if __name__ == "__main__":
    main()
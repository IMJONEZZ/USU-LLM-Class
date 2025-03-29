# from zenml import step
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
)


def evaluator():
    """Generates text using the trained LLaMA model with proper error handling and tokenization fixes."""

    # Load the trained LLaMA model and tokenizer
    model_path = "assn_6_llama"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    # Set pad_token to eos_token (LLaMA models don't have a default pad token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to GPU if available, otherwise use CPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.to(device)

    # Define the prompt
    prompt = "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?"

    # Tokenize input and ensure proper padding
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Check if input tokens exceed vocabulary size
    vocab_size = model.config.vocab_size
    if (inputs["input_ids"].cpu() >= vocab_size).any():
        raise ValueError("Input contains tokens outside the model's vocabulary!")

    model.eval()

    # Generate text with proper padding and attention mask
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
            temperature=0.7,  # Keep temperature low to avoid instability
            do_sample=False,  # Disable sampling for now
        )

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


print(evaluator())

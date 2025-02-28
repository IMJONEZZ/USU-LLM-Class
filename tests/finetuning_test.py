import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Define model name (Ensure it is accessible)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-step-50K"

@pytest.fixture(scope="module")
def load_model():
    """Load model and tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", load_in_8bit=True
    )
    return model, tokenizer

@pytest.fixture(scope="module")
def sample_dataset():
    """Create a small dataset for testing tokenization."""
    data = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    ]
    return Dataset.from_list(data)

def test_model_loading(load_model):
    """Check if model and tokenizer load successfully."""
    model, tokenizer = load_model
    assert model is not None, "Model failed to load."
    assert tokenizer is not None, "Tokenizer failed to load."

def test_tokenization(load_model, sample_dataset):
    """Test if tokenization works properly."""
    _, tokenizer = load_model

    def tokenize_function(examples):
        prompt = [f"Q: {q} A: {a}" for q, a in zip(examples["question"], examples["answer"])]
        return tokenizer(prompt, padding="max_length", truncation=True, max_length=256)

    tokenized_dataset = sample_dataset.map(tokenize_function, batched=True)
    
    assert "input_ids" in tokenized_dataset.column_names, "Tokenization failed!"
    assert len(tokenized_dataset["input_ids"][0]) > 0, "Empty tokenized input!"

def test_model_inference(load_model):
    """Test if model can generate coherent text."""
    model, tokenizer = load_model

    input_text = "Q: What is the capital of France? A:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        output = model.generate(input_ids, max_length=50)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(generated_text) > len(input_text), "Generated text is too short!"
    assert "Paris" in generated_text or "France" in generated_text, "Model output incorrect!"

def test_training_setup():
    """Ensure training parameters are properly configured."""
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./llama-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,  # For quick testing
        save_steps=100,
        logging_steps=10,
        optim="adamw_bnb_8bit"
    )

    assert training_args.per_device_train_batch_size > 0, "Batch size should be positive!"
    assert training_args.learning_rate > 0, "Learning rate should be positive!"
    assert training_args.num_train_epochs > 0, "Epochs should be positive!"


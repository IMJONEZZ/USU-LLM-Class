import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from zenml.pipelines import pipeline
from zenml.steps import step

# Define the model name
model_name = 'meta-llama/Llama-3.2-1B'

# Define the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the training step
@step
def train_model() -> AutoModelForCausalLM:
    # Load the tokenized dataset
    train_dataset = tokenized_datasets["train"]
    
    # Set the model to training mode
    model.train()
    
    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=5e-5)
    
    # Training loop
    for epoch in range(3):  # Train for 3 epochs
        for batch in train_dataset:
            inputs = torch.tensor(batch['input_ids'])
            labels = torch.tensor(batch['input_ids'])
            
            # Forward pass
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model

# Define the pipeline
@pipeline
def finetune_pipeline(train_model):
    train_model()

# Run the pipeline
pipeline_instance = finetune_pipeline(train_model=train_model)
pipeline_instance.run()
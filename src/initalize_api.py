from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face Hub
repo_id = "eliasthompson304/llama3_story_generator"

# Download and cache the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Move model to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize FastAPI
app = FastAPI()


# Define request structure
class InputText(BaseModel):
    prompt: str
    max_length: int = 100


# Endpoint for text generation
@app.post("/generate/")
async def generate_text(input: InputText):
    inputs = tokenizer(input.prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=input.max_length)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": generated_text}

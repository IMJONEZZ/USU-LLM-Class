import os
import json
import torch
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

# -------------------------------------------------
# 1. ENV Vars & Basic Config
# -------------------------------------------------
with open("keys.json") as f:
    keys = json.load(f)
HF_TOKEN = keys["huggingfaceToken"]
MODEL_NAME = os.environ.get("MODEL_NAME", "core42/jais-13b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# 2. Load LLM (Llama / Mistral / etc.)
# -------------------------------------------------
print("Loading tokenizer and model. This may take a moment...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    token=HF_TOKEN,
    trust_remote_code=True,
)
model.to(DEVICE)
print("Model loaded successfully.")

# -------------------------------------------------
# 3. Response Generation
# -------------------------------------------------
def generate_response(question: str) -> str:
    """
    Generates a response using iterative token generation with a progress bar.
    """
    # Tokenize with attention mask
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    
    # Determine how many new tokens to generate
    input_length = input_ids.shape[-1]
    max_new_tokens = 200 - input_length
    
    # Initialize output_ids with the input prompt
    output_ids = input_ids.clone()
    
    # Initialize progress bar
    progress_bar = tqdm(total=max_new_tokens, desc="Generating", leave=False)
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Generate one additional token
        outputs = model.generate(
            output_ids,
            attention_mask=attention_mask,
            max_length=output_ids.shape[-1] + 1,  # only add one token
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        # Extract the new token from the output
        new_token = outputs[0, -1].unsqueeze(0).unsqueeze(0)
        output_ids = torch.cat([output_ids, new_token], dim=-1)
        
        # Update the progress bar
        progress_bar.update(1)
        
        # Stop early if the EOS token is generated
        if new_token.item() == tokenizer.eos_token_id:
            break
    
    progress_bar.close()
    
    # Decode the generated tokens into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    response = response.split("\n")[0]
    return response


# -------------------------------------------------
# 4. FastAPI App Setup
# -------------------------------------------------
app = FastAPI()

class InferenceInput(BaseModel):
    inputs: str

@app.post("/predict", response_class=HTMLResponse)
async def predict(input_data: InferenceInput):
    """
    Inference endpoint.
    Expects JSON: {"inputs": "your question here"}
    Returns an HTML snippet with the question and answer wrapped in divs.
    """
    question = input_data.inputs
    answer = generate_response(question)
    
    # Create an HTML snippet with clearly separated parts for question and answer.
    html_content = f"""
    <div class="qa-entry">
      <div class="question"><strong>Question:</strong> {question}</div>
      <div class="answer"><strong>Answer:</strong> {answer}</div>
    </div>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

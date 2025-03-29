import os
import json
import torch
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------
# 1. ENV Vars & Basic Config
# -------------------------------------------------
with open("keys.json") as f:
    keys = json.load(f)
HF_TOKEN = keys["huggingfaceToken"]
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")

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
)
model.to(DEVICE)
print("Model loaded successfully.")

# -------------------------------------------------
# 3. Response Generation
# -------------------------------------------------
def generate_response(question: str) -> str:
    """
    Generates a response using the raw LLM without custom knowledge.
    """
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(DEVICE)

    output = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
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

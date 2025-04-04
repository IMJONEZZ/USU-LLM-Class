import os
import json
import torch
import uvicorn
import wikipedia  # For fetching Wikipedia summaries

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

# Set pad token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    token=HF_TOKEN,
)
model.to(DEVICE)
print("Model loaded successfully.")

# -------------------------------------------------
# 3. Wikipedia Agent Section
# -------------------------------------------------
class WikipediaAgent:
    """
    A simple Wikipedia agent that fetches a summary for a given query.
    """
    def __init__(self, sentences: int = 2):
        self.sentences = sentences  # Number of sentences to return in the summary

    def get_summary(self, query: str) -> str:
        try:
            # Use wikipedia.summary to get a brief summary
            summary = wikipedia.summary(query, sentences=self.sentences)
            return summary
        except Exception as e:
            return f"Error fetching Wikipedia summary: {e}"

# Instantiate the WikipediaAgent
wiki_agent = WikipediaAgent()

# -------------------------------------------------
# 4. Response Generation with Wikipedia Context
# -------------------------------------------------
def generate_response(question: str) -> str:
    """
    Generates a response using the raw LLM, enriched with context from Wikipedia.
    """
    # Get context from Wikipedia using the WikipediaAgent
    wiki_context = wiki_agent.get_summary(question)
    
    # Combine the Wikipedia context with the original question
    combined_prompt = f"Here is some context from Wikipedia:\n{wiki_context}\n\nQuestion: {question}\nAnswer:"
    
    # Tokenize with attention mask and truncation
    tokens = tokenizer(combined_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens.input_ids.to(DEVICE)
    attention_mask = tokens.attention_mask.to(DEVICE)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,  # Increased length to accommodate context and answer
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response

# -------------------------------------------------
# 5. FastAPI App Setup
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

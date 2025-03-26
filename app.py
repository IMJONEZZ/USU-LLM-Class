# app.py - FastAPI server for serving the fine-tuned Llama 3.2 model
import os
import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

from config import SYSTEM_PROMPT, MAX_NEW_TOKENS
from utils import extract_structured_answer

# Initialize FastAPI app
app = FastAPI(
    title="Llama 3.2 API Server",
    description="API server for serving the fine-tuned Llama 3.2 model",
    version="0.1.0",
)

# Global variables for model and tokenizer
global_model = None
global_tokenizer = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    question: str
    temperature: float = 0.7
    max_tokens: int = MAX_NEW_TOKENS
    system_prompt: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    full_response: str
    reasoning: Optional[str] = None
    answer: str


def load_model(model_path, hf_token=None):
    """Load the model and tokenizer."""
    # Get HF token from environment if not provided
    hf_token = hf_token or os.environ.get("HF_TOKEN")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
    )

    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        load_in_8bit=False,
        quantization_config=None,
    )

    return model, tokenizer


@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    global global_model, global_tokenizer

    # Default to llama_finetuned directory if it exists
    model_path = os.environ.get("MODEL_PATH", "llama_finetuned")
    hf_token = os.environ.get("HF_TOKEN")

    try:
        global_model, global_tokenizer = load_model(model_path, hf_token)
    except Exception as e:
        # Log the error but don't fail startup - we'll handle this at request time
        print(f"Error loading model at startup: {e}")


@app.get("/healthcheck")
async def healthcheck():
    """Healthcheck endpoint."""
    model_loaded = global_model is not None and global_tokenizer is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict endpoint for generating text from the model."""
    global global_model, global_tokenizer

    # Check if model is loaded
    if global_model is None or global_tokenizer is None:
        # Try loading model now if it wasn't loaded at startup
        try:
            model_path = os.environ.get("MODEL_PATH", "llama_finetuned")
            hf_token = os.environ.get("HF_TOKEN")
            global_model, global_tokenizer = load_model(model_path, hf_token)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    # Use provided system prompt or default
    system_prompt = request.system_prompt or SYSTEM_PROMPT

    # Generation parameters
    generation_params = {
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
    }

    # Format prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question},
    ]

    prompt = ""
    for msg in messages:
        prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

    # Tokenize
    inputs = global_tokenizer(prompt, return_tensors="pt").to(global_model.device)

    try:
        # Generate answer
        with torch.no_grad():
            outputs = global_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_params,
            )

        # Decode the generated output
        generated_text = global_tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        )

        # Extract reasoning and answer
        reasoning, answer = extract_structured_answer(generated_text)

        # If no structured answer found, use the whole text
        if not answer:
            answer = generated_text

        return PredictionResponse(
            full_response=generated_text, reasoning=reasoning, answer=answer
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating prediction: {str(e)}"
        )


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)

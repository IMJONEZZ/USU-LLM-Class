import os
import json
import torch
import uvicorn
import faiss
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# 1. ENV Vars & Basic Config
# -------------------------------------------------
# Read environment variables for the huggingface token and model name.

with open("keys.json") as f:
    keys = json.load(f)
HF_TOKEN = keys["huggingfaceToken"]
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")

# Decide if we use GPU or CPU
DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"

# -------------------------------------------------
# 2. Knowledge Base
# -------------------------------------------------
CUSTOM_KNOWLEDGE_BASE = {
    "opera_time": "If it takes 1 hour for 60 people to play an opera, adding more people does not change the time required. The performance length stays the same, so 600 people will also take 1 hour.",
    "feathers_vs_pound": "A British pound (currency) is a fixed unit of money, while a pound of feathers is a measure of weight. The pound of feathers is heavier.",
    "christmas_tree": "A tree in the living room with boxes underneath usually indicates Christmas morning, with presents placed under the tree.",
    "knuckle_cracking": "Cracking knuckles releases gas bubbles in the joints. It does not cause arthritis but may lead to reduced grip strength over time.",
    "shark_in_pool": "If there is a shark in the basement pool, it is irrelevant to going upstairs unless the shark has supernatural powers or can leave the water.",
    "woodchuck_riddle": "If only 5 pounds of wood exist in the world, a woodchuck could chuck at most 5 pounds of wood.",
    "us_president": "The current President of the United States is Joe Biden (as of 2024).",
    "talos_myth": "Talos is a figure from Greek mythology, a giant bronze automaton created to protect Crete.",
    "parallel_spelling": "The word 'parallel' contains 3 'L's.",
    "sphinx_riddle": "The riddle of the Sphinx: 'What walks on four legs in the morning, two in the afternoon, and three in the evening?' Answer: A human (crawling as a baby, walking as an adult, using a cane when old).",
}

# -------------------------------------------------
# 3. Load Embedding Model & Create FAISS Index
# -------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_texts = list(CUSTOM_KNOWLEDGE_BASE.values())
knowledge_keys = list(CUSTOM_KNOWLEDGE_BASE.keys())

# Encode knowledge base
embeddings = embedder.encode(knowledge_texts)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

id_to_text = {i: knowledge_texts[i] for i in range(len(knowledge_texts))}


def retrieve_knowledge(question, top_k=2, threshold=0.7) -> Optional[str]:
    """
    Returns the best matching knowledge snippet if the distance is
    below the given threshold; otherwise returns None.
    """
    question_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)

    best_match_id = indices[0][0]
    best_match_distance = distances[0][0]

    # Lower distance => more relevant
    if best_match_distance < threshold:
        return id_to_text[best_match_id]

    return None


# -------------------------------------------------
# 4. Load LLM (Llama / Mistral / etc.)
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
# 5. Response Generation
# -------------------------------------------------
def generate_response(question: str) -> str:
    """
    Attempt knowledge base retrieval first; if relevant info is found,
    return that (with a note). Otherwise, fall back to the LLM.
    """
    retrieved_knowledge = retrieve_knowledge(question)

    if retrieved_knowledge:
        return f"{retrieved_knowledge} (Retrieved from knowledge base.)"

    # If no relevant knowledge is found, use LLM
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(DEVICE)

    # We keep these generation args modest for demonstration:
    output = model.generate(
        input_ids,
        max_length=100,
        temperature=0.7,  # Adjust randomness
        top_p=0.9,  # Control diversity
        repetition_penalty=1.2,  # Combat repetition
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    # Sometimes models produce repeated lines. We'll do a simple safeguard:
    response = response.split("\n")[0]

    return response


# -------------------------------------------------
# 6. FastAPI App Setup
# -------------------------------------------------
app = FastAPI()


# Pydantic model for the inference input
class InferenceInput(BaseModel):
    inputs: str


@app.post("/predict")
async def predict(input_data: InferenceInput):
    """
    Hugging Face–style inference endpoint.
    Expects JSON: {"inputs": "your question here"}
    Returns: {"generated_text": "answer text"}
    """
    question = input_data.inputs
    answer = generate_response(question)
    return {"generated_text": answer}


# Serve the static front-end from the "static" folder.
# A simple way is to just serve index.html at root:
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    # For local dev: uvicorn app:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

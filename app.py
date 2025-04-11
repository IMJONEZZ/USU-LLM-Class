import streamlit as st
from fastapi import FastAPI
import os
import json
import torch
import wikipedia
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Basic Config & Model Loading ---- #
with open("keys.json") as f:
    keys = json.load(f)
HF_TOKEN = keys["huggingfaceToken"]
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.write("Loading tokenizer and model. Please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    token=HF_TOKEN,
)
model.to(DEVICE)
st.write("Model loaded successfully!")

# ---- Wikipedia Agent Functionality ---- #
class WikipediaAgent:
    """Fetch a brief summary from Wikipedia for a given query."""
    def __init__(self, sentences: int = 2):
        self.sentences = sentences

    def get_summary(self, query: str) -> str:
        try:
            summary = wikipedia.summary(query, sentences=self.sentences)
            return summary
        except Exception as e:
            return f"Error fetching Wikipedia summary: {e}"

wiki_agent = WikipediaAgent()

def generate_response(question: str) -> str:
    """
    Generates a response using both Wikipedia context and the LLM.
    """
    wiki_context = wiki_agent.get_summary(question)
    combined_prompt = (
        f"Here is some context from Wikipedia:\n{wiki_context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    tokens = tokenizer(
        combined_prompt, return_tensors="pt", padding=True, truncation=True
    )
    input_ids = tokens.input_ids.to(DEVICE)
    attention_mask = tokens.attention_mask.to(DEVICE)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,  # Adjust as needed
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response

# ---- Streamlit User Interface ---- #
st.title("LLM with Wikipedia Context")
st.write("Enter your question below to get an answer augmented with Wikipedia context.")

# Text input for the question
question = st.text_input("Your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Generating response..."):
            answer = generate_response(question)
        st.markdown("**Answer:**")
        st.write(answer)
    else:
        st.error("Please enter a question.")

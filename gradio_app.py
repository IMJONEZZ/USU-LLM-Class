import gradio as gr
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def chat_with_ollama(message, history):
    # Build full prompt from message history
    prompt = ""
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    prompt += f"User: {message}\nAssistant:"

    # Call Ollama
    response = requests.post(
        OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    result = response.json()["response"].strip()

    # Add assistant's reply to chat
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result})
    return history


gr.ChatInterface(fn=chat_with_ollama, type="messages").launch()

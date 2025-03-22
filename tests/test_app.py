import sys
import os
from fastapi.testclient import TestClient

# Add the repository root to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



# Import the FastAPI app from your main module
from app import app

client = TestClient(app)


def test_root_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Ask the Llama!" in response.text


def test_predict_with_known_question():
    question = "How many Ls are in the word 'parallel'?"
    payload = {"inputs": question}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    generated_text = data.get("generated_text", "")
    assert "Retrieved from knowledge base." in generated_text


def test_predict_with_unknown_question():
    question = "What is the capital of France?"
    payload = {"inputs": question}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    generated_text = data.get("generated_text", "")
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0

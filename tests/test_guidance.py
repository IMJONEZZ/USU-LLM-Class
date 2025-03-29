import sys
import os
from fastapi.testclient import TestClient

# Add the repository root to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the FastAPI app from your main module (assumed to be in app.py)
from app import app

client = TestClient(app)


def test_root_index():
    response = client.get("/")
    assert response.status_code == 200
    # Verify that the response is an HTML page and contains the expected title.
    assert "text/html" in response.headers["content-type"]
    assert "<title>Ask the Llama!" in response.text


def test_predict_with_known_question():
    question = "How many Ls are in the word 'parallel'?"
    payload = {"inputs": question}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Check that the response is HTML.
    assert "text/html" in response.headers["content-type"]
    html_content = response.text
    # Verify that the HTML snippet includes the expected div wrappers.
    assert '<div class="qa-entry">' in html_content
    assert '<div class="question">' in html_content
    assert '<div class="answer">' in html_content
    # Also check that the original question appears in the HTML.
    assert question in html_content


def test_predict_with_unknown_question():
    question = "What is the capital of France?"
    payload = {"inputs": question}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Verify that the returned content is HTML.
    assert "text/html" in response.headers["content-type"]
    html_content = response.text
    # Ensure the HTML contains the expected structure.
    assert '<div class="qa-entry">' in html_content
    assert '<div class="question">' in html_content
    assert '<div class="answer">' in html_content
    # Check that the question is present in the response.
    assert question in html_content

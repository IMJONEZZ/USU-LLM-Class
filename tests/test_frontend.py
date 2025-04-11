import torch

# Import the elements we want to test from your app
# app.py is one folder up from the test file, so we need to go up one level
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

# Define a fake get_summary to simulate WikipediaAgent behavior
def fake_get_summary(query):
    return "Dummy Wikipedia summary"

# Define a fake generate function for the model that returns a dummy tensor.
def fake_generate(*args, **kwargs):
    # The content of the tensor here is not used because we will override tokenizer.decode.
    # We just need any tensor with the proper shape.
    return torch.tensor([[0, 1, 2]])

# Define a fake decode function for the tokenizer that returns a string with the answer.
def fake_decode(tokens, skip_special_tokens=True):
    # The generate_response function expects the decoded string to include "Answer:" 
    # so that it splits on it and returns what follows.
    return "Answer: Dummy answer"

def test_generate_response(monkeypatch):
    # Override the WikipediaAgent.get_summary function
    monkeypatch.setattr(app.wiki_agent, "get_summary", fake_get_summary)
    
    # Override the model.generate function so we don't actually run the model
    monkeypatch.setattr(app.model, "generate", fake_generate)
    
    # Override the tokenizer.decode function to return a controlled string
    monkeypatch.setattr(app.tokenizer, "decode", fake_decode)
    
    # Define a sample question
    question = "What is AI?"
    
    # Call the function under test
    answer = app.generate_response(question)
    
    # Check if the returned answer is as expected (after splitting on "Answer:")
    assert answer == "Dummy answer", f"Expected 'Dummy answer' but got: {answer}"

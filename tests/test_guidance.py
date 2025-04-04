import pytest
from flask import Flask, jsonify
from app import app  # Assuming your Flask app is in a file named app.py
import json


# Test for the /generate route (Flask endpoint)
def test_generate_valid_prompt():
    # Valid prompt to test the generation functionality
    prompt = {"prompt": "Tell me a joke."}

    # Sending POST request to Flask
    response = app.test_client().post(
        "/generate", data=json.dumps(prompt), content_type="application/json"
    )

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if the response contains valid HTML
    assert b"<!DOCTYPE html>" in response.data
    assert b"<div>" in response.data
    assert b"</div>" in response.data


def test_generate_invalid_prompt():
    # Test with missing 'prompt' field
    invalid_prompt = {}

    response = app.test_client().post(
        "/generate", data=json.dumps(invalid_prompt), content_type="application/json"
    )

    # Check for a 400 error because the prompt is missing
    assert response.status_code == 400
    assert b'{"error": "No prompt provided!"}' in response.data


def test_generate_empty_prompt():
    # Test with an empty string as a prompt
    prompt = {"prompt": ""}

    response = app.test_client().post(
        "/generate", data=json.dumps(prompt), content_type="application/json"
    )

    # Check if the status code is 200 and that the generated response is not empty
    assert response.status_code == 200
    assert b"<div>" in response.data
    assert len(response.data) > 100  # Ensure the model output is not empty


# Test guidance logic directly (you could mock the guidance model or test specific parts here)
def test_generate_html_output():
    from app import generate_html_output

    prompt = "Describe the process of photosynthesis."
    html_output = generate_html_output(prompt)

    # Check if the HTML structure is returned properly
    assert "<html" in html_output
    assert "<body>" in html_output
    assert "<h1>Generated Response</h1>" in html_output
    assert prompt in html_output  # Ensure the prompt is included in the HTML

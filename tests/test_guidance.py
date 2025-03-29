import pytest
import json
import re
import os
from app import app

# Check if OpenAI API key is available
api_key_available = os.environ.get("OPENAI_API_KEY") is not None


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# Mark tests that require API calls
require_api = pytest.mark.skipif(
    not api_key_available,
    reason="OpenAI API key not available. Set OPENAI_API_KEY environment variable to run this test.",
)


@require_api
def test_generate_html_endpoint_article(client):
    """Test the /generate-html endpoint with article content type."""
    response = client.post(
        "/generate-html", json={"topic": "Machine Learning", "content_type": "article"}
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "html" in data
    html = data["html"]

    # Check for valid HTML structure
    assert re.search(r"<!DOCTYPE html>|<html", html, re.IGNORECASE)
    assert "</html>" in html

    # Check for required elements in article
    assert "<title>" in html and "</title>" in html
    assert ("<h1>" in html and "</h1>" in html) or (
        "<header>" in html and "</header>" in html
    )
    assert "<section>" in html or "<div" in html or "<article>" in html
    assert "<h2>" in html and "</h2>" in html
    assert "<p>" in html and "</p>" in html
    assert "<footer>" in html and "</footer>" in html


@require_api
def test_generate_html_endpoint_product(client):
    """Test the /generate-html endpoint with product content type."""
    response = client.post(
        "/generate-html",
        json={"topic": "Wireless Headphones", "content_type": "product"},
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "html" in data
    html = data["html"]

    # Check for valid HTML structure
    assert re.search(r"<!DOCTYPE html>|<html", html, re.IGNORECASE)
    assert "</html>" in html

    # Check for product-specific elements
    assert "<title>" in html and "</title>" in html
    assert any(tag in html for tag in ["<h1>", "<h2>", "<header>"])
    assert any(tag in html for tag in ["<ul>", "<ol>", "<div class"])
    assert "<li>" in html and "</li>" in html
    assert "$" in html or "price" in html.lower()  # Price indicator
    assert "button" in html.lower() or "buy" in html.lower()  # Buy button


def test_missing_topic(client):
    """Test error handling when topic is missing."""
    response = client.post("/generate-html", json={"content_type": "article"})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Topic is required" in data["error"]


def test_invalid_content_type(client):
    """Test error handling when content type is invalid."""
    response = client.post(
        "/generate-html", json={"topic": "Test Topic", "content_type": "invalid"}
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Invalid content_type" in data["error"]


def test_missing_data(client):
    """Test error handling when no data is provided."""
    response = client.post("/generate-html", json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "No data provided" in data["error"]

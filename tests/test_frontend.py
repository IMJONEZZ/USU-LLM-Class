from unittest.mock import patch
from hw11 import query_ollama


@patch("llama_chat.requests.post")
def test_mock_success(mock_post):
    mock_post.return_value.json.return_value = {"response": "Mocked reply"}

    reply = query_ollama("Mock test")
    assert reply == "Mocked reply"


@patch("llama_chat.requests.post")
def test_mock_error(mock_post):
    mock_post.return_value.json.return_value = {"error": "model not found"}

    reply = query_ollama("Test error")
    assert "⚠️" in reply or "Unexpected" in reply

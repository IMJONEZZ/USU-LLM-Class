import requests
import pytest

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3"


@pytest.fixture(scope="module")
def base_url():
    return OLLAMA_URL


@pytest.fixture(scope="module")
def model_name():
    return MODEL_NAME


def test_ollama_server_is_up(base_url):
    response = requests.get(base_url)
    assert response.status_code == 200, "Ollama server is not reachable."


def test_model_is_available(base_url, model_name):
    response = requests.get(f"{base_url}/api/tags")
    response.raise_for_status()
    models = response.json()["models"]
    model_names = [m["name"] for m in models]
    assert model_name in model_names, (
        f"Model '{model_name}' is not available. Found: {model_names}"
    )


def test_basic_chat_response(base_url, model_name):
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    response = requests.post(
        f"{base_url}/api/chat",
        json={"model": model_name, "messages": messages, "stream": False},
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    assert isinstance(content, str) and len(content) > 0, (
        "Chat response is empty or invalid."
    )


def test_multi_turn_chat_response(base_url, model_name):
    messages = [
        {"role": "user", "content": "What’s the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And Germany?"},
    ]
    response = requests.post(
        f"{base_url}/api/chat",
        json={"model": model_name, "messages": messages, "stream": False},
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    assert "Berlin" in content or "capital" in content.lower(), (
        "Unexpected chat response for Germany."
    )

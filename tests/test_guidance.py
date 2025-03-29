import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_html_response():
    response = client.get("/generate?character=John&setting=forest&genre=fantasy")
    assert response.status_code == 200
    assert "html" in response.headers["content-type"]
    assert "<html>" in response.text  # check if the response contains HTML

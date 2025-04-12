
import pytest
import streamlit as st
from unittest.mock import patch, MagicMock
import app  

@pytest.fixture(autouse=True)
def clear_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()

def test_session_state_initialization():
    assert "messages" in st.session_state
    assert isinstance(st.session_state.messages, list)

@patch("app.OpenAI")
def test_openai_stream_called(mock_openai):
    # Mock OpenAI response stream
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter(["Hello"])
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_stream
    mock_openai.return_value = mock_client

    # Simulate a user prompt
    st.session_state.messages = [{"role": "user", "content": "Hi!"}]
    
    # Simulate calling the chat logic
    with patch("streamlit.chat_input", return_value="Hi again"):
        with patch("streamlit.write_stream", return_value="Hi response"):
            # Importing app will run its script logic
            import importlib
            importlib.reload(app)

    # Make sure OpenAI API was called
    mock_client.chat.completions.create.assert_called_once()

import sqlite3
import pytest
import ollama
import frontend


# Fixture to set up an in-memory SQLite database for testing.
@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    # Create a temporary SQLite file for this test session.
    db_path = tmp_path / "test_chat_history.db"
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME
        )
    """)
    connection.commit()

    # Monkey-patch the global connection and cursor in the module.
    monkeypatch.setattr(frontend, "conn", connection)
    monkeypatch.setattr(frontend, "c", cursor)
    yield connection
    connection.close()


def clear_db():
    """Helper to clear the database."""
    frontend.c.execute("DELETE FROM chats")
    frontend.conn.commit()


def test_save_and_load_history():
    """Test that messages are saved to and loaded from the database."""
    clear_db()
    frontend.save_message("user", "Hello")
    frontend.save_message("assistant", "Hi there")
    history = frontend.load_history()
    # load_history returns a list of dicts (each dict: {"role":..., "content":...})
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there"


def test_user_function():
    """
    Test that when a user submits a message using the `user()` function,
    it returns an empty textbox output and updates the history with a user message
    and an empty assistant placeholder.
    """
    initial_history = []
    user_message = "Hi, bot!"
    out_msg, new_history = frontend.user(user_message, initial_history)

    # The message box should be cleared.
    assert out_msg == ""
    # The new history should have two entries.
    assert len(new_history) == 2
    # First entry is the user message.
    assert new_history[0]["role"] == "user"
    assert new_history[0]["content"] == user_message
    # Second entry is a placeholder for the assistant.
    assert new_history[1]["role"] == "assistant"
    assert new_history[1]["content"] == ""


def fake_ollama_chat(model, messages, stream):
    """
    A fake streaming response to simulate the assistant's output.
    This generator yields tokens one by one.
    """
    for token in ["Hello", " ", "world", "!"]:
        yield {"message": {"content": token}}


def test_bot_function(monkeypatch):
    """
    Test that the `bot()` function correctly updates the assistant message
    using the streaming tokens produced by `chat_stream()`.
    """
    # Initialize a conversation with one user message and an empty assistant placeholder.
    history = [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": ""},
    ]
    # Monkey-patch the ollama.chat call with our fake streaming function.
    monkeypatch.setattr(ollama, "chat", fake_ollama_chat)

    # Collect all updates from the bot generator.
    final_history = None
    for updated_history in frontend.bot(history):
        final_history = updated_history

    # Ensure that the assistant's response is as expected.
    assert final_history is not None
    assert final_history[-1]["role"] == "assistant"
    assert final_history[-1]["content"] == "Hello world!"

import ollama
import gradio as gr
import sqlite3
from datetime import datetime
import threading
import subprocess

# Set up a threading lock for SQLite access.
db_lock = threading.Lock()
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
c = conn.cursor()

# Create table if it does not exist.
c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        timestamp DATETIME
    )
""")
conn.commit()


def save_message(role, content):
    with db_lock:
        c.execute(
            "INSERT INTO chats (role, content, timestamp) VALUES (?,?,?)",
            (role, content, datetime.now()),
        )
        conn.commit()


def load_history():
    """Load all saved messages from the database and return as a list of dicts."""
    with db_lock:
        c.execute("SELECT role, content FROM chats ORDER BY timestamp")
        rows = c.fetchall()
    # Return a list of dictionaries as expected by the Chatbot component.
    history = [{"role": row[0], "content": row[1]} for row in rows]
    return history


def get_model_list():
    """
    Run `ollama list` and parse the plain text output.
    The first line is assumed to be a header, so we skip it.
    Each subsequent line's first column is taken as the model name.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        # Skip header line if there is one.
        if len(lines) > 1:
            lines = lines[1:]
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                # First token in the line is the model name
                models.append(parts[0])
        return models if models else ["llama3.3:latest"]
    except Exception as e:
        print("Error fetching model list:", e)
        return ["llama3.3:latest"]


def chat_stream(user_message, history, model):
    """
    Send the conversation history to Ollama with streaming enabled using the selected model.
    Yields tokens as they arrive.
    """
    # Save the user message persistently.
    save_message("user", user_message)
    # Stream from ollama using the given model.
    response = ollama.chat(
        model=model,
        messages=history,
        stream=True,
    )
    full_response = ""
    for chunk in response:
        content = chunk["message"]["content"]
        full_response += content
        yield content
    # Save assistant response persistently.
    save_message("assistant", full_response)


# Gradio UI Setup.
with gr.Blocks() as demo:
    gr.Markdown("# Llama3.3 Chat with Persistent History & Model Selection")

    # Model selection dropdown
    model_dd = gr.Dropdown(
        label="Select Model", choices=get_model_list(), value="llama3.3:latest"
    )

    # Chatbot uses a list of dictionaries (each with 'role' and 'content').
    chatbot = gr.Chatbot(height=500, type="messages", value=load_history())
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear History")

    def user(user_message, history):
        """
        On message submission, append a new dict for the user message and a placeholder
        for the upcoming assistant message.
        """
        history = history or []
        history = history.copy()
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ""})
        return "", history

    def bot(history, model):
        """
        Stream and update the assistant response:
          - The user message is the second-to-last message.
          - The assistant's response is updated token-by-token in the last message.
        """
        if not history or len(history) < 2:
            yield history
            return

        user_message = history[-2]["content"]
        # Exclude the empty assistant placeholder from the history sent to the model.
        for token in chat_stream(user_message, history[:-1], model):
            history[-1]["content"] += token
            yield history

    # Wire up submission: first update state with the user's message, then stream the bot reply.
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, model_dd], [chatbot]
    )

    def clear_history():
        """Clear persistent history from both database and UI."""
        with db_lock:
            c.execute("DELETE FROM chats")
            conn.commit()
        return []

    clear.click(clear_history, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()

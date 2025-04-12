import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent import WebAgent


# test if the web search actually gets web info
def test_web_search_returns_results():
    agent = WebAgent()
    result = agent.search_web("current president of the United States")
    assert "http" in result or "https" in result, "Web search didn't return any URLs"


# test if memory is actually stored or not
def test_memory_remembering():
    agent = WebAgent()
    session_id = "test_memory"
    agent.remember(session_id, "Hello?", "Hi there!")
    assert session_id in agent.memory
    assert agent.memory[session_id][0] == ("Hello?", "Hi there!")

import os
import shutil
import pytest
from prompt_engineering import engineer_prompt


@pytest.fixture(autouse=True)
def cleanup():
    # Remove any previously saved dspy_program directory before and after tests
    if os.path.exists("./dspy_program/"):
        shutil.rmtree("./dspy_program/")
    yield
    if os.path.exists("./dspy_program/"):
        shutil.rmtree("./dspy_program/")


def test_engineer_prompt_returns_result():
    result = engineer_prompt()
    assert result is not None, "Expected a non-null result from engineer_prompt()"

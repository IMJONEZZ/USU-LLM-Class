import pytest
from main import hello_world


def test_hello_world():
    assert hello_world("World") == "Hello, World!"
    assert hello_world("Bob") == "Hello, Bob!"
    assert hello_world("") == "Hello, !"
    assert hello_world(123) == "Hello, 123!"
    with pytest.raises(TypeError):
        hello_world()

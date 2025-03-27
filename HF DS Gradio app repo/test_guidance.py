import pytest
from guidance_helper import generate_html
from bs4 import BeautifulSoup

def test_generate_html():
    """Test if the Guidance-based function generates valid HTML."""
    prompt = "The Future of AI"
    output = generate_html(prompt)

    # Check if output is a string
    assert isinstance(output, str), "Output should be a string."

    # Parse with BeautifulSoup to check if it's valid HTML
    soup = BeautifulSoup(output, "html.parser")

    # Check for expected HTML elements
    assert soup.find("h1"), "Missing <h1> tag in output."
    assert soup.find("p"), "Missing <p> tag in output."
    assert soup.find("ul"), "Missing <ul> tag in output."

    # Ensure the title matches
    assert soup.title.string == prompt, f"Expected title '{prompt}', but got '{soup.title.string}'."

    print("✅ Test passed: Generated HTML is valid and structured correctly.")

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main()
from generate_html import template, generator
from bs4 import BeautifulSoup

def test_html_structure():
    student = generator("Give me a student description", seed=789001)
    output = template.render(student=student.dict())
    # Verify basic HTML structure
    assert "<!DOCTYPE html>" in output
    assert "<html>" in output
    assert "</html>" in output
    
    # Parse with BeautifulSoup for deeper validation
    soup = BeautifulSoup(output, "html.parser")
    assert soup.title.string == "Student Profile"
    
def test_student_data_presence():
    # Generate test data
    student = generator("Give me a student description", seed=789001)
    
    # Fix Pydantic deprecation warning
    html_output = template.render(student=student.model_dump())  # Changed from .dict()
    soup = BeautifulSoup(html_output, "html.parser")

    # Check for presence of data in HTML structure
    h1 = soup.find("h1")
    assert h1 is not None, "Missing h1 element"
    assert h1.text.startswith("Student Profile:"), "Invalid h1 content"

    # Check list items using CSS selectors
    age_item = soup.select_one("li:has(> strong:-soup-contains('Age:'))")
    assert age_item is not None, "Missing Age field"
    assert age_item.text.strip().startswith("Age:"), "Invalid Age format"

    major_item = soup.select_one("li:has(> strong:-soup-contains('Major:'))")
    assert major_item is not None, "Missing Major field"

    # Verify GPA formatting
    gpa_item = soup.select_one("li:has(> strong:-soup-contains('GPA:'))")
    assert gpa_item is not None, "Missing GPA field"
    gpa_value = float(gpa_item.text.split(":")[1].strip())
    assert 0 <= gpa_value <= 4.0, "GPA out of valid range"

    # Check clubs list structure
    clubs_list = soup.select("li:has(> strong:-soup-contains('Clubs Joined:')) ul")
    assert len(clubs_list) == 1, "Missing clubs sublist"
    assert len(clubs_list[0].find_all("li")) >= 1, "No clubs listed"


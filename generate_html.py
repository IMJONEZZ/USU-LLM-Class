from pydantic import BaseModel, constr, confloat
from typing import List
import outlines
from jinja2 import Template

# Define the Student Pydantic model
class Student(BaseModel):
    name: constr(max_length=20)
    age: int
    major: str
    minor: str
    gpa: confloat(ge=0, le=4)  # GPA should be between 0 and 4
    clubs: List[str]

model = outlines.models.transformers(
    "meta-llama/Llama-3.1-8B",
    model_kwargs={"device_map": "auto"}  # Adjust device settings as needed
)

template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Profile</title>
</head>
<body>
    <h1>Student Profile: {{ student.name }}</h1>
    <ul>
        <li><strong>Age:</strong> {{ student.age }}</li>
        <li><strong>Major:</strong> {{ student.major }}</li>
        <li><strong>Minor:</strong> {{ student.minor }}</li>
        <li><strong>GPA:</strong> {{ student.gpa }}</li>
        <li><strong>Clubs Joined:</strong>
            <ul>
            {% for club in student.clubs %}
                <li>{{ club }}</li>
            {% endfor %}
            </ul>
        </li>
    </ul>
</body>
</html>
"""

template = Template(template_str)
generator = outlines.generate.json(model, Student)
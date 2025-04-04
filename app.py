# app.py
from flask import Flask, request, jsonify
import guidance
from guidance import models, gen, select

# Load the Falcon model using guidance
falcon = models.Transformers("tiiuae/falcon-rw-1b")

# Initialize Flask app
app = Flask(__name__)


# Function to generate HTML from the model response
def generate_html_output(prompt):
    lm = falcon + prompt + gen(max_tokens=100)

    # Create a simple HTML structure to return the model output
    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM HTML Output</title>
    </head>
    <body>
        <h1>Generated Response</h1>
        <div>{lm}</div>
    </body>
    </html>
    """
    return html_output


@app.route("/generate", methods=["POST"])
def generate():
    # Get the JSON data from the request
    data = request.get_json()
    prompt = data.get("prompt")

    if prompt:
        html_response = generate_html_output(prompt)
        return html_response
    else:
        return jsonify({"error": "No prompt provided!"}), 400


if __name__ == "__main__":
    app.run(debug=True)

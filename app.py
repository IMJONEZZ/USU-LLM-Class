from flask import Flask, request, jsonify
from guidance import models, gen
from guidance import user, assistant, system
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def get_api_key():
    """Get the OpenAI API key from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found! Please set the OPENAI_API_KEY environment variable."
        )
    return api_key


@app.route("/generate-html", methods=["POST"])
def generate_html():
    try:
        # Get API key from environment variable
        api_key = get_api_key()

        # Initialize OpenAI model with guidance
        llm = models.OpenAI("gpt-3.5-turbo", api_key=api_key)

        # Get parameters from request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        topic = data.get("topic", "")
        content_type = data.get("content_type", "article")

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        # Validate content_type
        valid_content_types = ["article", "product"]
        if content_type not in valid_content_types:
            return jsonify(
                {
                    "error": f"Invalid content_type. Must be one of: {', '.join(valid_content_types)}"
                }
            ), 400

        lm = llm

        with system():
            lm += f"You are a helpful assistant that generates valid HTML content about {topic}."

        with user():
            if content_type == "article":
                lm += f"""
                Please create an HTML article page about {topic} with the following structure:
                1. A descriptive title and meta tag
                2. Two main sections with headings and paragraphs
                3. A footer

                The HTML should be properly formatted and valid. Only return the HTML code without any explanation.
                """
            else:  # product
                lm += f"""
                Please create an HTML product page about {topic} with the following structure:
                1. A descriptive title and meta tag
                2. Product details section with at least 3 features in a list
                3. A price display and buy button
                4. A footer

                The HTML should be properly formatted and valid. Only return the HTML code without any explanation.
                """

        with assistant():
            lm += gen(name="html_output")

        # Extract the HTML content from the result
        html_content = lm["html_output"]

        # Return the generated HTML
        return jsonify({"html": html_content})

    except ValueError as ve:
        logger.error(f"API Key Error: {str(ve)}")
        return jsonify({"error": str(ve)}), 401
    except Exception as e:
        logger.error(f"Error generating HTML: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

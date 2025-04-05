from flask import Flask, request, jsonify
from prompt_optimizer import PromptOptimizer
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

        # Initialize the PromptOptimizer with the specified model
        optimizer = PromptOptimizer("gpt-3.5-turbo", api_key=api_key)

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

        # Generate HTML content using the PromptOptimizer
        html_content = optimizer.generate_html(topic, content_type)

        # Return the generated HTML
        return jsonify({"html": html_content})

    except ValueError as ve:
        logger.error(f"API Key Error: {str(ve)}")
        return jsonify({"error": str(ve)}), 401
    except Exception as e:
        logger.error(f"Error generating HTML: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/structured-output", methods=["POST"])
def structured_output():
    try:
        # Get API key from environment variable
        api_key = get_api_key()

        # Initialize the PromptOptimizer with the specified model
        optimizer = PromptOptimizer("gpt-3.5-turbo", api_key=api_key)

        # Get parameters from request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        topic = data.get("topic", "")
        output_format = data.get("format", "json")

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        # Generate structured output using the PromptOptimizer
        result = optimizer.generate_structured_output(topic, output_format)

        # Return the generated output
        return jsonify({"result": result})

    except ValueError as ve:
        logger.error(f"API Key Error: {str(ve)}")
        return jsonify({"error": str(ve)}), 401
    except Exception as e:
        logger.error(f"Error generating structured output: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

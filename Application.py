from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Initialize Hugging Face Gradio client
client = Client(
    "Jashonnew/deepseek-ai-DeepSeek-R1",
    hf_token="your_hf_token_here",
)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Send message to Hugging Face model
    try:
        result = client.predict(user_input, api_name="/apply")
        print(f"API Response: {result}")
        return jsonify({"response": result})
    except Exception as e:
        print(f"Gradio API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

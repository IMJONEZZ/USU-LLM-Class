import gradio as gr
import requests


def query_ollama(prompt):
    try:
        response = requests.post(
            "http://172.29.16.1:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
        )
        data = response.json()
        return data.get("response", "⚠️ Unexpected structure.")
    except Exception as e:
        return f"❌ Exception: {e}"


# import json  # Add this if not already imported

# def query_ollama(prompt):
#     try:
#         response = requests.post(
#             "http://172.29.16.1:11434/api/generate",
#             json={
#                 "model": "llama3.2",
#                 "prompt": prompt,
#                 "stream": False
#             }
#         )
#         data = response.json()
#         print("RAW JSON:\n", json.dumps(data, indent=2))  # Print prettified JSON
#         if "response" in data:
#             return data["response"]
#         elif "error" in data:
#             return f"⚠️ Ollama Error: {data['error']}"
#         else:
#             return f"⚠️ Unexpected structure. Full response:\n{json.dumps(data, indent=2)}"
#     except Exception as e:
#         return f"❌ Exception: {e}"


# Launch the Gradio UI
gr.Interface(
    fn=query_ollama, inputs="text", outputs="text", title="Chat with LLaMA 3 (Ollama)"
).launch()

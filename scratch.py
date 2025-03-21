from gradio_client import Client

client = Client("Jashonnew/deepseek-ai-DeepSeek-R1")

response = client.predict(
    "Hello, how are you?", api_name="/apply"
)  # Using the working endpoint
print(f"API Response: {response}")

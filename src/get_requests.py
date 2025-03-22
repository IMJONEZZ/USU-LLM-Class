import requests

url = "http://localhost:8000/generate"  # Change to your Hugging Face API URL if hosted there

params = {"character": "Alice", "setting": "a futuristic city", "genre": "sci-fi"}

response = requests.get(url, params=params)
print(response.json()["story"])

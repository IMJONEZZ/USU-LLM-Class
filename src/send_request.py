import requests

url = "http://0.0.0.0:8001"

params = {"character": "Lucy", "setting": "the beach", "genre": "mystery"}

response = requests.get(url, params=params)

if response.status_code == 200:
    print(response.text)
else:
    print(f"Error: {response.status_code}")

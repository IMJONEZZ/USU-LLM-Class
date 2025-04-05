import requests

url = "http://127.0.0.1:8001/ask"  # use 127.0.0.1 instead of 0.0.0.0 for local requests

params = {"question": "", "session_id": ""}

response = requests.get(url, params=params)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")

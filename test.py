import requests

url = "http://127.0.0.1:8000/chat"

data = {
    "message": "Apa kabar?"
}

res = requests.post(url, json=data)

print("STATUS:", res.status_code)
print("RAW TEXT:", res.text)
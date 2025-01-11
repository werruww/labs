import requests
import json
# Define the request parameters
url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2",
    "prompt": "Explain quantum computing in simple terms",
    "temperature": 0.5,
    "max_tokens": 150
   
}

# Send the request
response = requests.post(url, json=payload)
final_response = ""
for chunk in response.text.split("\n"):
    if chunk.strip():  # Skip empty lines
        data = json.loads(chunk)
        final_response += data.get("response", "")
# Print the response
print(final_response)

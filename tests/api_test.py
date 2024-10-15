import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API token from environment variables
api_token = os.getenv("LICHESS_API_TOKEN")

# Define the FEN and multiPv parameters
fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
multi_pv = 1

# Define the API endpoint and parameters
url = os.getenv("LICHESS_CLOUD_ENGINE")
params = {"fen": fen, "multiPv": multi_pv}

# Define the headers
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

# Send the GET request
response = requests.get(url, headers=headers, params=params)

# Check the response status and print the result
if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print("Error:", response.status_code, response.text)

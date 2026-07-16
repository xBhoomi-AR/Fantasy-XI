import requests
import json

url = "https://fantasy.premierleague.com/api/element-summary/1/"

data = requests.get(url).json()

print(data["history"][0].keys())
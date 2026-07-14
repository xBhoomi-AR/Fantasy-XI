import requests
import json
from pathlib import Path

url = "https://fantasy.premierleague.com/api/fixtures/"

data = requests.get(url).json()

Path("data/raw/fpl").mkdir(parents=True, exist_ok=True)

with open("data/raw/fpl/fixtures.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"Downloaded {len(data)} fixtures")
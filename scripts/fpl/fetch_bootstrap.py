import requests
import json
from pathlib import Path

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

response = requests.get(URL, timeout=30)
response.raise_for_status()

data = response.json()

output_dir = Path("data/raw/fpl")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "bootstrap_static.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Saved to:", output_file)
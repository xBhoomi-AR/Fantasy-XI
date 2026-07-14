import json
from pprint import pprint

with open("data/raw/fpl/bootstrap_static.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total Players: {len(data['elements'])}")

print("\nFirst Player:\n")

pprint(data["elements"][0])
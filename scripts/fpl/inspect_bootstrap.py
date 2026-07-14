import json

with open("data/raw/fpl/bootstrap_static.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTop Level Keys:")
print(data.keys())

print("\n")

for key, value in data.items():
    if isinstance(value, list):
        print(f"{key:20} -> List ({len(value)})")
    elif isinstance(value, dict):
        print(f"{key:20} -> Dict ({len(value)} keys)")
    else:
        print(f"{key:20} -> {type(value)}")
import json
import time
from pathlib import Path

import requests

# Load player ids

with open("data/raw/fpl/bootstrap_static.json", encoding="utf-8") as f:
    bootstrap = json.load(f)

players = bootstrap["elements"]

save_dir = Path("data/raw/fpl/element_summary")
save_dir.mkdir(parents=True, exist_ok=True)

downloaded = 0

for player in players:

    player_id = player["id"]

    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"

    try:
        response = requests.get(url, timeout=15)

        if response.status_code == 200:

            with open(
                save_dir / f"{player_id}.json",
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(response.json(), f, indent=4)

            downloaded += 1

            print(f"{player_id} Yes")

        else:

            print(f"{player_id} failed")

    except Exception:

        print(f"{player_id} error")

    time.sleep(0.1)

print()
print("Downloaded", downloaded, "player histories")
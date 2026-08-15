import numpy as np
import pandas as pd
from pathlib import Path


# Keeping the seed fixed makes the example dataset reproducible
np.random.seed(42)

NUM_GAMEWEEKS = 10

# Small player pool for the RL prototype
players = [
    {"player_id": "P01", "name": "Arjun",  "position": "DEF"},
    {"player_id": "P02", "name": "Kabir",  "position": "DEF"},
    {"player_id": "P03", "name": "Rohan",  "position": "DEF"},
    {"player_id": "P04", "name": "Dev",    "position": "DEF"},

    {"player_id": "P05", "name": "Ayaan",  "position": "MID"},
    {"player_id": "P06", "name": "Vihaan", "position": "MID"},
    {"player_id": "P07", "name": "Reyansh","position": "MID"},
    {"player_id": "P08", "name": "Aditya", "position": "MID"},

    {"player_id": "P09", "name": "Ishaan", "position": "FWD"},
    {"player_id": "P10", "name": "Aryan",  "position": "FWD"},
    {"player_id": "P11", "name": "Dhruv",  "position": "FWD"},
    {"player_id": "P12", "name": "Kunal",  "position": "FWD"},
]


def create_players():
    player_data = []

    for player in players:
        # Ability is hidden from the RL agent and is only used to simulate performance
        ability = np.random.uniform(0.45, 0.90)

        # Better players generally cost more, with a little variation
        price = 4.0 + ability * 6.0 + np.random.normal(0, 0.35)
        price = round(np.clip(price, 4.0, 10.0) * 2) / 2

        player_data.append({
            **player,
            "price": price,
            "base_ability": round(ability, 3)
        })

    return player_data


def generate_season(seed=None):
    if seed is not None:
        np.random.seed(seed)

    season_players = create_players()
    rows = []

    for player in season_players:
        # Every player starts with a reasonable level of form
        form = np.random.uniform(0.40, 0.80)

        for gameweek in range(1, NUM_GAMEWEEKS + 1):

            # Form changes gradually rather than jumping randomly every week
            form += np.random.normal(0, 0.08)
            form = np.clip(form, 0.20, 1.00)

            # FPL-style fixture difficulty from 1 to 5
            fixture_difficulty = np.random.randint(1, 6)

            # Easier fixtures give a positive effect, difficult ones give a negative effect
            fixture_effect = (3 - fixture_difficulty) * 0.45

            # Underlying expected performance for this gameweek
            expected_points = (
                1.0
                + player["base_ability"] * 5.0
                + form * 3.0
                + fixture_effect
            )

            # Predicted points imitate the output we would eventually get from Phase II
            prediction_noise = np.random.normal(0, 0.65)
            predicted_points = expected_points + prediction_noise
            predicted_points = round(max(0.0, predicted_points), 2)

            # Actual performance has separate and slightly larger uncertainty
            performance_noise = np.random.normal(0, 0.90)
            actual_points = expected_points + performance_noise
            actual_points = int(round(max(0.0, actual_points)))

            rows.append({
                "gameweek": gameweek,
                "player_id": player["player_id"],
                "name": player["name"],
                "position": player["position"],
                "price": player["price"],
                "form": round(float(form), 3),
                "fixture_difficulty": fixture_difficulty,
                "predicted_points": predicted_points,
                "actual_points": actual_points
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_season(seed=42)

    # Save the example season beside this script
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / "synthetic_fpl_data.csv"
    df.to_csv(output_path, index=False)

    print("\nSynthetic FPL dataset created successfully.")
    print(f"Rows: {len(df)}")
    print(f"Players: {df['player_id'].nunique()}")
    print(f"Gameweeks: {df['gameweek'].nunique()}")

    print("\nFirst 20 rows:")
    print(df.head(20).to_string(index=False))

    print("\nAverage predicted points:", round(df["predicted_points"].mean(), 2))
    print("Average actual points:", round(df["actual_points"].mean(), 2))

    correlation = df["predicted_points"].corr(df["actual_points"])
    print("Predicted vs actual correlation:", round(correlation, 3))

    print(f"\nSaved to: {output_path}")
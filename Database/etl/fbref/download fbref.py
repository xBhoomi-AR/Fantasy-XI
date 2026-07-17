import os
import pandas as pd
import soccerdata as sd



CACHE_DIR = "../../cache/fbref"

os.makedirs(CACHE_DIR, exist_ok=True)



SEASONS = [
    "2023-2024",
    "2024-2025",
    "2025-2026"
]



def download_team_match_logs():

    print("\nInitializing FBRef...")

    fb = sd.FBref(
        leagues="ENG-Premier League",
        seasons=SEASONS
    )

    print("Downloading Team Match Logs...\n")

    team_logs = fb.read_team_match_stats(stat_type="summary")

    output_file = os.path.join(
        CACHE_DIR,
        "team_match_logs.pkl"
    )

    team_logs.to_pickle(output_file)

    print("======================================")
    print("Download Complete")
    print(f"Rows     : {len(team_logs)}")
    print(f"Columns  : {len(team_logs.columns)}")
    print(f"Saved to : {output_file}")
    print("======================================")

    return team_logs


def main():

    team_logs = download_team_match_logs()

    print("\nFirst Five Rows:\n")
    print(team_logs.head())

    print("\nColumns:\n")
    print(team_logs.columns.tolist())


if __name__ == "__main__":
    main()
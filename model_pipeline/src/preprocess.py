import pandas as pd
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "player_match_stats.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "player_gameweek.csv"


# Columns that are useful for predicting future points
STAT_COLUMNS = [
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
    "saves",
    "recoveries",
    "tackles",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "npxg",
    "xg",
    "xa",
    "xgi",
]


def load_data():
    print("Loading raw data...")

    df = pd.read_csv(RAW_FILE)

    print(f"Raw shape: {df.shape}")

    return df


def clean_data(df):
    print("Cleaning data...")

    # Make sure the columns used below are numeric
    for column in STAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce")

    # These values mean that the player did not record that statistic.
    # For our historical features, treating them as zero is more useful
    # than leaving them as missing values.
    for column in STAT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    # Remove rows where the basic player/gameweek information is missing.
    df = df.dropna(
        subset=["player_id", "season", "gameweek"]
    )

    return df


def aggregate_gameweeks(df):
    print("Combining fixture data into player Gameweeks...")

    # A player can have more than one fixture in the same Gameweek.
    # We need one row per player per Gameweek for the sequence model.
    group_columns = [
        "player_id",
        "team_id",
        "season",
        "gameweek",
        "position",
    ]

    available_stats = [
        column for column in STAT_COLUMNS
        if column in df.columns
    ]

    aggregation = {}

    for column in available_stats:
        aggregation[column] = "sum"

    # Home/away can be different for a double Gameweek,
    # so we keep the average instead of pretending there was one value.
    if "was_home" in df.columns:
        aggregation["was_home"] = "mean"

    gameweek_df = (
        df.groupby(group_columns, as_index=False)
        .agg(aggregation)
    )

    return gameweek_df


def create_target(df):
    print("Creating next Gameweek target...")

    # Sort before shifting so that each player's previous Gameweeks
    # appear in the correct order.
    df = df.sort_values(
        ["player_id", "season", "gameweek"]
    ).reset_index(drop=True)

    # The model will use previous Gameweeks to predict the next one.
    df["target_points"] = (
        df.groupby(["player_id", "season"])["total_points"]
        .shift(-1)
    )

    # The final Gameweek of each season has no next Gameweek,
    # so it cannot be used as a supervised training example.
    df = df.dropna(subset=["target_points"]).copy()

    df["target_points"] = df["target_points"].astype(float)

    return df


def save_data(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed shape: {df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")


def main():
    df = load_data()

    df = clean_data(df)

    df = aggregate_gameweeks(df)

    df = create_target(df)

    save_data(df)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path


# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "merged_gw.csv"
OUTPUT_PATH = BASE_DIR / "data" / "xgboost_prepared.csv"


# Load the original gameweek data
df = pd.read_csv(DATA_PATH)

print("\n--- ORIGINAL DATASET ---")
print("Shape:", df.shape)
print("Unique players:", df["element"].nunique())
print("GW range:", df["GW"].min(), "-", df["GW"].max())


# Aggregate multiple fixtures for the same player in a gameweek.
# This handles double gameweeks since the target is total FPL points per GW.
sum_columns = [
    "total_points",
    "minutes",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "goals_conceded",
    "goals_scored",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "yellow_cards",
    "creativity",
    "influence",
    "threat",
    "ict_index"
]

# Use only the columns available in the dataset
sum_columns = [col for col in sum_columns if col in df.columns]

first_columns = [
    "name",
    "position",
    "team"
]

first_columns = [col for col in first_columns if col in df.columns]

aggregation_rules = {}

for col in sum_columns:
    aggregation_rules[col] = "sum"

for col in first_columns:
    aggregation_rules[col] = "first"

gameweek_df = (
    df.groupby(["element", "GW"], as_index=False)
    .agg(aggregation_rules)
)

print("\n--- AFTER PLAYER-GAMEWEEK AGGREGATION ---")
print("Shape:", gameweek_df.shape)

# Check that each player has only one row per gameweek
duplicate_count = (
    gameweek_df
    .groupby(["element", "GW"])
    .size()
    .gt(1)
    .sum()
)

print(
    "Duplicate player-gameweek combinations:",
    duplicate_count
)


# Sort each player's history before creating time-based features
gameweek_df = gameweek_df.sort_values(
    ["element", "GW"]
).reset_index(drop=True)


# Create features from the player's previous recorded gameweek.
# Current gameweek values are not used as model inputs.
lag_columns = [
    "total_points",
    "minutes",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "creativity",
    "influence",
    "threat",
    "ict_index"
]

lag_columns = [
    col for col in lag_columns
    if col in gameweek_df.columns
]

for column in lag_columns:
    gameweek_df[f"{column}_lag1"] = (
        gameweek_df
        .groupby("element")[column]
        .shift(1)
    )


# Create rolling averages using only previous gameweeks.
# shift(1) prevents the current gameweek from leaking into its own features.
rolling_columns = [
    "total_points",
    "minutes",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index"
]

rolling_columns = [
    col for col in rolling_columns
    if col in gameweek_df.columns
]

for column in rolling_columns:

    gameweek_df[f"{column}_rolling3"] = (
        gameweek_df
        .groupby("element")[column]
        .transform(
            lambda x: x.shift(1)
            .rolling(
                window=3,
                min_periods=3
            )
            .mean()
        )
    )

    gameweek_df[f"{column}_rolling5"] = (
        gameweek_df
        .groupby("element")[column]
        .transform(
            lambda x: x.shift(1)
            .rolling(
                window=5,
                min_periods=5
            )
            .mean()
        )
    )


# Points scored in the current gameweek are the prediction target
gameweek_df["target_points"] = gameweek_df["total_points"]


# Remove early rows where there is not enough player history
model_df = gameweek_df.dropna(
    subset=[
        "total_points_rolling5",
        "minutes_rolling5"
    ]
).copy()


# Final checks
print("\n--- PREPARED ML DATASET ---")
print("Shape:", model_df.shape)

print(
    "GW range:",
    model_df["GW"].min(),
    "-",
    model_df["GW"].max()
)

print(
    "Unique players:",
    model_df["element"].nunique()
)

final_duplicates = (
    model_df
    .groupby(["element", "GW"])
    .size()
)

final_duplicates = final_duplicates[
    final_duplicates > 1
]

print(
    "Remaining duplicate player-GW combinations:",
    len(final_duplicates)
)


# Show a small sample of the prepared features
display_columns = [
    "name",
    "GW",
    "total_points_lag1",
    "total_points_rolling3",
    "total_points_rolling5",
    "minutes_rolling3",
    "expected_goals_rolling3",
    "expected_assists_rolling3",
    "target_points"
]

display_columns = [
    col for col in display_columns
    if col in model_df.columns
]

print("\n--- EXAMPLE PREPARED DATA ---")

print(
    model_df[
        display_columns
    ].head(20)
)


# Save the dataset used by the models
model_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nPrepared dataset saved to:")
print(OUTPUT_PATH)
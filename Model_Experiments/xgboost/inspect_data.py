import pandas as pd
from pathlib import Path


# Dataset path
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "merged_gw.csv"
)


# Load the original dataset
df = pd.read_csv(DATA_PATH)


# General dataset information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# Gameweek and player coverage
print("\n--- GAMEWEEK RANGE ---")
print("Minimum GW:", df["GW"].min())
print("Maximum GW:", df["GW"].max())

print("\n--- NUMBER OF UNIQUE PLAYERS ---")
print(df["element"].nunique())


# Target variable summary
print("\n--- TOTAL POINTS SUMMARY ---")
print(df["total_points"].describe())


# Check how many rows correspond to players who actually appeared
print("\n--- PLAYING TIME ANALYSIS ---")

total_rows = len(df)
zero_minute_rows = (df["minutes"] == 0).sum()
played_rows = (df["minutes"] > 0).sum()

zero_minute_percentage = (
    zero_minute_rows
    / total_rows
    * 100
)

print("Total rows:", total_rows)
print("Rows with 0 minutes:", zero_minute_rows)
print("Rows with > 0 minutes:", played_rows)

print(
    "Percentage with 0 minutes:",
    round(zero_minute_percentage, 2),
    "%"
)


# Point distribution for players who did not appear
print("\n--- POINTS WHEN PLAYER DID NOT PLAY ---")

print(
    df[
        df["minutes"] == 0
    ]["total_points"].describe()
)
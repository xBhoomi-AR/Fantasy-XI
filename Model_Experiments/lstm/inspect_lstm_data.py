import pandas as pd


# Load the original merged gameweek dataset
df = pd.read_csv("Model_Experiments/data/merged_gw.csv")


print("\nDataset shape:")
print(df.shape)


print("\nAll columns:")
for column in df.columns:
    print(column)


print("\nFirst 5 rows:")
print(df.head())


print("\nDataset information:")
df.info()


print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))


print("\nDuplicate rows:")
print(df.duplicated().sum())


print("\nGameweek range:")
print("Minimum GW:", df["GW"].min())
print("Maximum GW:", df["GW"].max())


print("\nUnique players:")
print("By name:", df["name"].nunique())
print("By element:", df["element"].nunique())


print("\nRows per gameweek:")
print(df.groupby("GW").size())


print("\nPlayer positions:")
print(df["position"].value_counts(dropna=False))


print("\nTotal points distribution:")
print(df["total_points"].describe())


print("\nZero-point rows:")
zero_points = (df["total_points"] == 0).sum()
print("Count:", zero_points)
print("Percentage:", (zero_points / len(df)) * 100)


print("\nZero-minute rows:")
zero_minutes = (df["minutes"] == 0).sum()
print("Count:", zero_minutes)
print("Percentage:", (zero_minutes / len(df)) * 100)


print("\nPlayers with the most gameweek records:")
print(df.groupby(["element", "name"]).size().sort_values(ascending=False).head(20))


print("\nPlayers with the fewest gameweek records:")
print(df.groupby(["element", "name"]).size().sort_values().head(20))


print("\nDuplicate player-gameweek combinations:")
player_gw_duplicates = df.duplicated(subset=["element", "GW"]).sum()
print(player_gw_duplicates)


print("\nExample player gameweek history:")
example_player = df["element"].iloc[0]

example_history = df[df["element"] == example_player][
    ["name", "element", "GW", "minutes", "total_points"]
].sort_values("GW")

print(example_history)

print("\nInvestigating duplicate player-gameweek rows:")

duplicate_player_gw = df[
    df.duplicated(subset=["element", "GW"], keep=False)
].sort_values(["element", "GW"])

print("Total rows involved:")
print(len(duplicate_player_gw))

print("\nNumber of unique players involved:")
print(duplicate_player_gw["element"].nunique())

print("\nGameweeks containing duplicate player-GW rows:")
print(
    duplicate_player_gw.groupby("GW")
    .size()
    .sort_index()
)

print("\nFirst 30 duplicate player-GW rows:")
print(
    duplicate_player_gw[
        [
            "name",
            "element",
            "GW",
            "fixture",
            "opponent_team",
            "was_home",
            "minutes",
            "total_points"
        ]
    ].head(30).to_string(index=False)
)


print("\nExact duplicate rows:")
exact_duplicates = df[df.duplicated(keep=False)]

print(
    exact_duplicates[
        [
            "name",
            "element",
            "GW",
            "fixture",
            "opponent_team",
            "was_home",
            "minutes",
            "total_points"
        ]
    ].sort_values(["element", "GW"]).to_string(index=False)
)
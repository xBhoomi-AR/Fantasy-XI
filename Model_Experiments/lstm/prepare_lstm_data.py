import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "merged_gw.csv"


# Number of previous gameweeks used to predict the next gameweek
SEQUENCE_LENGTH = 5


# Load the original dataset
df = pd.read_csv(DATA_PATH)

print("\n--- ORIGINAL DATASET ---")
print("Shape:", df.shape)


# Remove exact duplicate rows
duplicate_count = df.duplicated().sum()
print("Exact duplicate rows:", duplicate_count)

df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)


# Features that should be added together during double gameweeks
sum_columns = [
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "creativity",
    "threat",
    "influence",
    "ict_index",
    "bonus",
    "bps",
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
    "saves",
    "penalties_saved",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed"
]


# Keep only columns that actually exist in the dataset
sum_columns = [
    column for column in sum_columns
    if column in df.columns
]


# These values identify the player
first_columns = [
    "name",
    "position",
    "team"
]

first_columns = [
    column for column in first_columns
    if column in df.columns
]


# Define how each column should be combined
aggregation_rules = {}

for column in sum_columns:
    aggregation_rules[column] = "sum"

for column in first_columns:
    aggregation_rules[column] = "first"


# Combine multiple fixtures from the same gameweek
gameweek_df = (
    df.groupby(
        ["element", "GW"],
        as_index=False
    )
    .agg(aggregation_rules)
)


print("\n--- AFTER PLAYER-GAMEWEEK AGGREGATION ---")
print("Shape:", gameweek_df.shape)

duplicate_player_gw = gameweek_df.duplicated(
    subset=["element", "GW"]
).sum()

print(
    "Duplicate player-gameweek rows:",
    duplicate_player_gw
)


# Sort every player's history chronologically
gameweek_df = gameweek_df.sort_values(
    ["element", "GW"]
).reset_index(drop=True)


# Convert position into numerical columns
gameweek_df = pd.get_dummies(
    gameweek_df,
    columns=["position"],
    prefix="position",
    dtype=int
)


# Features available at each historical timestep
feature_columns = [
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "creativity",
    "threat",
    "influence",
    "ict_index",
    "bonus",
    "bps",
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
    "saves",
    "penalties_saved",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "position_GK",
    "position_DEF",
    "position_MID",
    "position_FWD"
]


# Keep only features available in the prepared data
feature_columns = [
    column for column in feature_columns
    if column in gameweek_df.columns
]


print("\n--- LSTM FEATURES ---")
print("Number of features:", len(feature_columns))

for feature in feature_columns:
    print("-", feature)


# Store the sequences and their metadata
sequences = []
targets = []
metadata = []


# Create sequences separately for every player
for element, player_df in gameweek_df.groupby("element"):

    player_df = player_df.sort_values(
        "GW"
    ).reset_index(drop=True)

    # Start only when the player has enough previous gameweeks
    for i in range(SEQUENCE_LENGTH, len(player_df)):

        history = player_df.iloc[
            i - SEQUENCE_LENGTH:i
        ]

        target_row = player_df.iloc[i]

        # Make sure the sequence contains consecutive gameweeks
        expected_gws = list(
            range(
                int(target_row["GW"]) - SEQUENCE_LENGTH,
                int(target_row["GW"])
            )
        )

        actual_gws = history["GW"].astype(int).tolist()

        if actual_gws != expected_gws:
            continue

        sequence = history[
            feature_columns
        ].values.astype(np.float32)

        target = float(
            target_row["total_points"]
        )

        sequences.append(sequence)
        targets.append(target)

        metadata.append({
            "element": element,
            "name": target_row["name"],
            "target_gw": int(target_row["GW"]),
            "target_minutes": target_row["minutes"],
            "target_points": target
        })


# Convert sequences and targets into NumPy arrays
X = np.array(
    sequences,
    dtype=np.float32
)

y = np.array(
    targets,
    dtype=np.float32
)

metadata_df = pd.DataFrame(metadata)


print("\n--- CREATED LSTM SEQUENCES ---")
print("Number of sequences:", len(X))
print("Input shape:", X.shape)
print("Target shape:", y.shape)


# Split sequences based on the target gameweek
train_mask = metadata_df["target_gw"] <= 29

val_mask = (
    (metadata_df["target_gw"] >= 30)
    & (metadata_df["target_gw"] <= 33)
)

test_mask = metadata_df["target_gw"] >= 34


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]


train_metadata = metadata_df[
    train_mask
].reset_index(drop=True)

val_metadata = metadata_df[
    val_mask
].reset_index(drop=True)

test_metadata = metadata_df[
    test_mask
].reset_index(drop=True)


print("\n--- CHRONOLOGICAL SPLIT ---")

print(
    "Train:",
    X_train.shape,
    "| Target GW",
    train_metadata["target_gw"].min(),
    "-",
    train_metadata["target_gw"].max()
)

print(
    "Validation:",
    X_val.shape,
    "| Target GW",
    val_metadata["target_gw"].min(),
    "-",
    val_metadata["target_gw"].max()
)

print(
    "Test:",
    X_test.shape,
    "| Target GW",
    test_metadata["target_gw"].min(),
    "-",
    test_metadata["target_gw"].max()
)


# Show a few sequences so we can manually verify them
print("\n--- EXAMPLE SEQUENCES ---")

for example_index in range(
    min(5, len(metadata_df))
):

    example = metadata_df.iloc[
        example_index
    ]

    target_gw = int(
        example["target_gw"]
    )

    input_gws = list(
        range(
            target_gw - SEQUENCE_LENGTH,
            target_gw
        )
    )

    print("\nPlayer:", example["name"])
    print("Element:", example["element"])
    print("Input GWs:", input_gws)
    print("Target GW:", target_gw)
    print("Target points:", example["target_points"])
    print(
        "Sequence shape:",
        X[example_index].shape
    )


    # Create folders for saved LSTM data and results
LSTM_DATA_DIR = BASE_DIR / "data" / "lstm_prepared"
RESULTS_DIR = BASE_DIR / "results"

LSTM_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Fit the scaler using training data only
scaler = StandardScaler()

num_features = X_train.shape[2]

X_train_2d = X_train.reshape(
    -1,
    num_features
)

scaler.fit(X_train_2d)


# Scale each dataset using the training scaler
X_train_scaled = scaler.transform(
    X_train.reshape(-1, num_features)
).reshape(X_train.shape)

X_val_scaled = scaler.transform(
    X_val.reshape(-1, num_features)
).reshape(X_val.shape)

X_test_scaled = scaler.transform(
    X_test.reshape(-1, num_features)
).reshape(X_test.shape)


# Convert back to float32 for PyTorch
X_train_scaled = X_train_scaled.astype(
    np.float32
)

X_val_scaled = X_val_scaled.astype(
    np.float32
)

X_test_scaled = X_test_scaled.astype(
    np.float32
)


print("\n--- SCALING COMPLETE ---")
print("Scaler fitted on training data only")

print(
    "Train shape:",
    X_train_scaled.shape
)

print(
    "Validation shape:",
    X_val_scaled.shape
)

print(
    "Test shape:",
    X_test_scaled.shape
)


# Save the prepared arrays
np.save(
    LSTM_DATA_DIR / "X_train.npy",
    X_train_scaled
)

np.save(
    LSTM_DATA_DIR / "y_train.npy",
    y_train
)

np.save(
    LSTM_DATA_DIR / "X_val.npy",
    X_val_scaled
)

np.save(
    LSTM_DATA_DIR / "y_val.npy",
    y_val
)

np.save(
    LSTM_DATA_DIR / "X_test.npy",
    X_test_scaled
)

np.save(
    LSTM_DATA_DIR / "y_test.npy",
    y_test
)


# Save metadata for later evaluation
train_metadata.to_csv(
    LSTM_DATA_DIR / "train_metadata.csv",
    index=False
)

val_metadata.to_csv(
    LSTM_DATA_DIR / "val_metadata.csv",
    index=False
)

test_metadata.to_csv(
    LSTM_DATA_DIR / "test_metadata.csv",
    index=False
)


# Save the fitted scaler
joblib.dump(
    scaler,
    LSTM_DATA_DIR / "lstm_scaler.pkl"
)


# Save feature names
with open(
    LSTM_DATA_DIR / "feature_names.txt",
    "w"
) as file:

    for feature in feature_columns:
        file.write(feature + "\n")


print("\n--- FILES SAVED ---")

print(
    "Prepared data folder:",
    LSTM_DATA_DIR
)

print(
    "Saved training, validation and test arrays"
)

print(
    "Saved metadata, scaler and feature names"
)
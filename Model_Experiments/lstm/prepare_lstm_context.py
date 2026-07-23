import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "merged_gw.csv"

OUTPUT_DIR = BASE_DIR / "data" / "lstm_context_prepared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 5


df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()

print("Dataset shape:", df.shape)


# Stats from previous gameweeks
history_features = [
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

history_features = [
    col for col in history_features
    if col in df.columns
]


# Add together performance from both fixtures in a double GW
aggregation = {
    col: "sum"
    for col in history_features
}

for col in ["name", "position", "team"]:
    if col in df.columns:
        aggregation[col] = "first"


gameweek_df = (
    df.groupby(["element", "GW"], as_index=False)
      .agg(aggregation)
      .sort_values(["element", "GW"])
      .reset_index(drop=True)
)


# Position is known information
gameweek_df = pd.get_dummies(
    gameweek_df,
    columns=["position"],
    prefix="position",
    dtype=int
)

position_cols = [
    col for col in gameweek_df.columns
    if col.startswith("position_")
]

history_features += position_cols


# Build target-GW fixture context separately
teams = sorted(df["team"].unique())
opponents = sorted(df["opponent_team"].unique())

context_rows = []

for (element, gw), group in df.groupby(["element", "GW"]):

    row = {
        "element": element,
        "GW": gw,

        # Player price at the start/current state of the GW
        "value": float(group["value"].iloc[0]),

        # Number of fixtures: 1 normally, 2 for a DGW
        "fixture_count": len(group),

        # One if at least one fixture is at home
        "has_home_fixture": int(group["was_home"].astype(bool).any()),

        # One if at least one fixture is away
        "has_away_fixture": int((~group["was_home"].astype(bool)).any())
    }

    team = group["team"].iloc[0]

    for current_team in teams:
        row[f"team_{current_team}"] = int(
            team == current_team
        )

    current_opponents = set(group["opponent_team"].tolist())

    for opponent in opponents:
        row[f"opponent_{opponent}"] = int(
            opponent in current_opponents
        )

    context_rows.append(row)


context_df = pd.DataFrame(context_rows)

context_feature_cols = [
    col for col in context_df.columns
    if col not in ["element", "GW"]
]


print("Historical features:", len(history_features))
print("Context features:", len(context_feature_cols))


sequences = []
contexts = []
targets = []
metadata = []


for element, player_df in gameweek_df.groupby("element"):

    player_df = (
        player_df.sort_values("GW")
                 .reset_index(drop=True)
    )

    for i in range(SEQUENCE_LENGTH, len(player_df)):

        history = player_df.iloc[
            i - SEQUENCE_LENGTH:i
        ]

        target_row = player_df.iloc[i]
        target_gw = int(target_row["GW"])

        expected_gws = list(
            range(
                target_gw - SEQUENCE_LENGTH,
                target_gw
            )
        )

        actual_gws = (
            history["GW"]
            .astype(int)
            .tolist()
        )

        if actual_gws != expected_gws:
            continue


        # Find context belonging to the GW being predicted
        target_context = context_df[
            (context_df["element"] == element)
            & (context_df["GW"] == target_gw)
        ]

        if len(target_context) != 1:
            continue


        sequence = (
            history[history_features]
            .values
            .astype(np.float32)
        )

        context = (
            target_context[context_feature_cols]
            .iloc[0]
            .values
            .astype(np.float32)
        )

        target = float(
            target_row["total_points"]
        )


        sequences.append(sequence)
        contexts.append(context)
        targets.append(target)

        metadata.append({
            "element": element,
            "name": target_row["name"],
            "target_gw": target_gw,
            "target_minutes": target_row["minutes"],
            "target_points": target
        })


X_seq = np.array(sequences, dtype=np.float32)
X_context = np.array(contexts, dtype=np.float32)
y = np.array(targets, dtype=np.float32)

metadata_df = pd.DataFrame(metadata)


print("\nSequences:", X_seq.shape)
print("Context:", X_context.shape)
print("Targets:", y.shape)


# Same chronological split as our original LSTM
train_mask = metadata_df["target_gw"] <= 29

val_mask = (
    (metadata_df["target_gw"] >= 30)
    & (metadata_df["target_gw"] <= 33)
)

test_mask = metadata_df["target_gw"] >= 34


X_seq_train = X_seq[train_mask]
X_seq_val = X_seq[val_mask]
X_seq_test = X_seq[test_mask]

X_context_train = X_context[train_mask]
X_context_val = X_context[val_mask]
X_context_test = X_context[test_mask]

y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]


train_metadata = metadata_df[train_mask].reset_index(drop=True)
val_metadata = metadata_df[val_mask].reset_index(drop=True)
test_metadata = metadata_df[test_mask].reset_index(drop=True)


# Scale historical features using training data only
seq_scaler = StandardScaler()

num_history_features = X_seq_train.shape[2]

seq_scaler.fit(
    X_seq_train.reshape(
        -1,
        num_history_features
    )
)


def scale_sequences(data):
    return seq_scaler.transform(
        data.reshape(-1, num_history_features)
    ).reshape(data.shape).astype(np.float32)


X_seq_train = scale_sequences(X_seq_train)
X_seq_val = scale_sequences(X_seq_val)
X_seq_test = scale_sequences(X_seq_test)


# Only value and fixture count need scaling.
# One-hot and binary variables should stay 0/1.
context_scaler = StandardScaler()

numeric_context = [
    context_feature_cols.index("value"),
    context_feature_cols.index("fixture_count")
]

context_scaler.fit(
    X_context_train[:, numeric_context]
)


def scale_context(data):

    data = data.copy()

    data[:, numeric_context] = context_scaler.transform(
        data[:, numeric_context]
    )

    return data.astype(np.float32)


X_context_train = scale_context(X_context_train)
X_context_val = scale_context(X_context_val)
X_context_test = scale_context(X_context_test)


# Save arrays
np.save(OUTPUT_DIR / "X_seq_train.npy", X_seq_train)
np.save(OUTPUT_DIR / "X_seq_val.npy", X_seq_val)
np.save(OUTPUT_DIR / "X_seq_test.npy", X_seq_test)

np.save(OUTPUT_DIR / "X_context_train.npy", X_context_train)
np.save(OUTPUT_DIR / "X_context_val.npy", X_context_val)
np.save(OUTPUT_DIR / "X_context_test.npy", X_context_test)

np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "y_test.npy", y_test)


train_metadata.to_csv(
    OUTPUT_DIR / "train_metadata.csv",
    index=False
)

val_metadata.to_csv(
    OUTPUT_DIR / "val_metadata.csv",
    index=False
)

test_metadata.to_csv(
    OUTPUT_DIR / "test_metadata.csv",
    index=False
)


joblib.dump(
    seq_scaler,
    OUTPUT_DIR / "sequence_scaler.pkl"
)

joblib.dump(
    context_scaler,
    OUTPUT_DIR / "context_scaler.pkl"
)


with open(
    OUTPUT_DIR / "history_features.txt",
    "w"
) as file:
    file.write("\n".join(history_features))


with open(
    OUTPUT_DIR / "context_features.txt",
    "w"
) as file:
    file.write("\n".join(context_feature_cols))


print("\n--- DONE ---")

print("Train sequence:", X_seq_train.shape)
print("Train context:", X_context_train.shape)

print("Validation sequence:", X_seq_val.shape)
print("Validation context:", X_context_val.shape)

print("Test sequence:", X_seq_test.shape)
print("Test context:", X_context_test.shape)

print("\nSaved to:", OUTPUT_DIR)
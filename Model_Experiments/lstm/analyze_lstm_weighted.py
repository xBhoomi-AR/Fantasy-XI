from pathlib import Path

import numpy as np
import pandas as pd


# Paths for results and prediction file
BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

PREDICTIONS_FILE = (
    RESULTS_DIR / "lstm_weighted_predictions.csv"
)


# Load the weighted LSTM predictions
df = pd.read_csv(PREDICTIONS_FILE)

print("\n--- LSTM PREDICTION ANALYSIS ---")

print("Number of predictions:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# Check how the actual scores are distributed
print("\n--- ACTUAL SCORE DISTRIBUTION ---")

score_groups = {
    "0-2": df["actual_points"].between(0, 2),
    "3-5": df["actual_points"].between(3, 5),
    "6-9": df["actual_points"].between(6, 9),
    "10+": df["actual_points"] >= 10
}

for label, mask in score_groups.items():

    count = mask.sum()

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{label:<5} "
        f"Count: {count:<5} "
        f"Percentage: {percentage:.2f}%"
    )


# Check the range and spread of model predictions
print("\n--- PREDICTION DISTRIBUTION ---")

predictions = df["predicted_points"]

print(
    f"Minimum: {predictions.min():.3f}"
)

print(
    f"Maximum: {predictions.max():.3f}"
)

print(
    f"Mean:    {predictions.mean():.3f}"
)

print(
    f"Median:  {predictions.median():.3f}"
)

print("\nPrediction percentiles:")

for percentile in [
    50,
    75,
    90,
    95,
    99
]:

    value = np.percentile(
        predictions,
        percentile
    )

    print(
        f"{percentile}th percentile: "
        f"{value:.3f}"
    )


# Compare average prediction with actual points for each score group
print("\n--- PREDICTION BIAS BY ACTUAL SCORE ---")

for label, mask in score_groups.items():

    group = df[mask]

    if len(group) == 0:
        continue

    actual_mean = (
        group["actual_points"].mean()
    )

    predicted_mean = (
        group["predicted_points"].mean()
    )

    bias = (
        predicted_mean
        - actual_mean
    )

    print(
        f"{label:<5} "
        f"Actual mean: {actual_mean:.3f} | "
        f"Predicted mean: {predicted_mean:.3f} | "
        f"Bias: {bias:.3f}"
    )


# Check MAE and RMSE separately for every test gameweek
print("\n--- PERFORMANCE BY GAMEWEEK ---")

gw_column = "target_gw"

for gw in sorted(
    df[gw_column].unique()
):

    group = df[
        df[gw_column] == gw
    ]

    errors = (
        group["predicted_points"]
        - group["actual_points"]
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    print(
        f"GW {gw:<2} | "
        f"Samples: {len(group):<4} | "
        f"MAE: {mae:.3f} | "
        f"RMSE: {rmse:.3f}"
    )


# Check how closely predictions follow actual scores
print("\n--- PREDICTION CORRELATION ---")

correlation = df[
    [
        "actual_points",
        "predicted_points"
    ]
].corr().iloc[0, 1]

print(
    f"Actual vs predicted correlation: "
    f"{correlation:.4f}"
)


# Compare the actual and predicted top 15 players in each GW
print("\n--- TOP PLAYER RANKING ANALYSIS ---")

TOP_N = 15

total_actual_top = 0
total_overlap = 0

for gw in sorted(
    df[gw_column].unique()
):

    group = df[
        df[gw_column] == gw
    ]

    actual_top = group.nlargest(
        TOP_N,
        "actual_points"
    )

    predicted_top = group.nlargest(
        TOP_N,
        "predicted_points"
    )

    actual_ids = set(
        actual_top["element"]
    )

    predicted_ids = set(
        predicted_top["element"]
    )

    overlap = len(
        actual_ids.intersection(
            predicted_ids
        )
    )

    total_actual_top += len(
        actual_ids
    )

    total_overlap += overlap

    print(
        f"GW {gw:<2} | "
        f"Top-{TOP_N} overlap: "
        f"{overlap}/{TOP_N}"
    )


overall_top_recall = (
    total_overlap
    / total_actual_top
)

print(
    f"\nOverall Top-{TOP_N} recall: "
    f"{overall_top_recall:.3f}"
)


# Check how the model performs on players who actually scored 6+ points
print("\n--- HIGH SCORER ANALYSIS ---")

high_scorers = df[
    df["actual_points"] >= 6
]

print(
    "Actual 6+ point performances:",
    len(high_scorers)
)

print(
    "Average actual score:",
    round(
        high_scorers[
            "actual_points"
        ].mean(),
        3
    )
)

print(
    "Average predicted score:",
    round(
        high_scorers[
            "predicted_points"
        ].mean(),
        3
    )
)

high_scorer_underprediction = (
    high_scorers["actual_points"]
    - high_scorers["predicted_points"]
).mean()

print(
    "Average underprediction:",
    round(
        high_scorer_underprediction,
        3
    )
)


# Show the players given the highest predictions by the model
print("\n--- HIGHEST MODEL PREDICTIONS ---")

columns_to_show = [
    column
    for column in [
        "name",
        "element",
        "target_gw",
        "actual_points",
        "predicted_points"
    ]
    if column in df.columns
]

print(
    df.nlargest(
        20,
        "predicted_points"
    )[columns_to_show]
    .to_string(index=False)
)


# Save MAE and RMSE for each gameweek
gw_summary = []

for gw in sorted(
    df[gw_column].unique()
):

    group = df[
        df[gw_column] == gw
    ]

    errors = (
        group["predicted_points"]
        - group["actual_points"]
    )

    gw_summary.append({
        "GW": gw,
        "samples": len(group),
        "MAE": np.mean(
            np.abs(errors)
        ),
        "RMSE": np.sqrt(
            np.mean(errors ** 2)
        )
    })


gw_summary = pd.DataFrame(
    gw_summary
)

summary_path = (
    RESULTS_DIR
    / "lstm_weighted_gw_analysis.csv"
)

gw_summary.to_csv(
    summary_path,
    index=False
)

print(
    "\nGameweek analysis saved to:",
    summary_path
)


# Check how many target players actually played in that GW
print("\n--- MINUTES / AVAILABILITY ANALYSIS ---")

minute_groups = {
    "0 minutes": df["target_minutes"] == 0,
    "1-29 minutes": df["target_minutes"].between(1, 29),
    "30-59 minutes": df["target_minutes"].between(30, 59),
    "60+ minutes": df["target_minutes"] >= 60
}

for label, mask in minute_groups.items():

    group = df[mask]

    if len(group) == 0:
        continue

    print(
        f"{label:<15} "
        f"Count: {len(group):<5} "
        f"Percentage: {len(group) / len(df) * 100:.2f}% | "
        f"Avg points: {group['actual_points'].mean():.3f}"
    )


# Look at score distribution only for players who played 60+ minutes
print("\n--- SCORE DISTRIBUTION FOR 60+ MINUTE PLAYERS ---")

played_60 = df[
    df["target_minutes"] >= 60
]

print(
    "Number of 60+ minute performances:",
    len(played_60)
)

for label, mask in {
    "0-2": played_60["actual_points"].between(0, 2),
    "3-5": played_60["actual_points"].between(3, 5),
    "6-9": played_60["actual_points"].between(6, 9),
    "10+": played_60["actual_points"] >= 10
}.items():

    count = mask.sum()

    print(
        f"{label:<5} "
        f"Count: {count:<5} "
        f"Percentage: {count / len(played_60) * 100:.2f}%"
    )


# Check top 15 ranking only among players who played 60+ minutes
print("\n--- TOP-15 RANKING FOR 60+ MINUTE PLAYERS ---")

total_overlap = 0
total_players = 0

for gw in sorted(df["target_gw"].unique()):

    gw_data = df[
        (df["target_gw"] == gw)
        & (df["target_minutes"] >= 60)
    ]

    actual_top = gw_data.nlargest(
        15,
        "actual_points"
    )

    predicted_top = gw_data.nlargest(
        15,
        "predicted_points"
    )

    actual_ids = set(actual_top["element"])
    predicted_ids = set(predicted_top["element"])

    overlap = len(
        actual_ids & predicted_ids
    )

    total_overlap += overlap
    total_players += len(actual_ids)

    print(
        f"GW {gw} | "
        f"Top-15 overlap: {overlap}/15"
    )


recall = total_overlap / total_players

print(
    f"\nOverall Top-15 recall "
    f"(60+ minute players): {recall:.3f}"
)
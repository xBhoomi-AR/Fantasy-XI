import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "lstm_context_prepared"
RESULTS_DIR = BASE_DIR / "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# load test data
X_seq = np.load(DATA_DIR / "X_seq_test.npy")
X_context = np.load(DATA_DIR / "X_context_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

metadata = pd.read_csv(DATA_DIR / "test_metadata.csv")


class ContextLSTM(nn.Module):
    def __init__(self, history_size, context_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=history_size,
            hidden_size=64,
            batch_first=True
        )

        self.fc1 = nn.Linear(64 + context_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, sequence, context):
        output, _ = self.lstm(sequence)

        history = output[:, -1, :]

        combined = torch.cat(
            (history, context),
            dim=1
        )

        combined = torch.relu(self.fc1(combined))

        return self.fc2(combined).squeeze(1)


model = ContextLSTM(
    X_seq.shape[2],
    X_context.shape[1]
).to(device)

model.load_state_dict(
    torch.load(
        RESULTS_DIR / "best_lstm_context_model.pt",
        map_location=device
    )
)

model.eval()

print("Best context model loaded.")


# prediction
seq_tensor = torch.tensor(
    X_seq,
    dtype=torch.float32
).to(device)

context_tensor = torch.tensor(
    X_context,
    dtype=torch.float32
).to(device)

with torch.no_grad():
    predictions = model(
        seq_tensor,
        context_tensor
    ).cpu().numpy()


# overall metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\n--- TEST RESULTS ---")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


# prediction dataframe
results = metadata.copy()

results["actual_points"] = y_test
results["predicted_points"] = predictions
results["absolute_error"] = np.abs(
    results["actual_points"] -
    results["predicted_points"]
)

results.to_csv(
    RESULTS_DIR / "lstm_context_predictions.csv",
    index=False
)


# score-group performance
print("\n--- PERFORMANCE BY ACTUAL POINTS ---")

groups = [
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-9", 6, 9),
    ("10+", 10, float("inf"))
]

for name, low, high in groups:

    group = results[
        (results["actual_points"] >= low) &
        (results["actual_points"] <= high)
    ]

    if len(group) == 0:
        continue

    group_mae = group["absolute_error"].mean()

    group_rmse = np.sqrt(
        np.mean(
            (
                group["actual_points"] -
                group["predicted_points"]
            ) ** 2
        )
    )

    print(
        f"{name:4s} | "
        f"Count: {len(group):4d} | "
        f"MAE: {group_mae:.4f} | "
        f"RMSE: {group_rmse:.4f}"
    )


# high scorer behaviour
high = results[
    results["actual_points"] >= 6
]

print("\n--- HIGH SCORER ANALYSIS ---")
print("6+ performances:", len(high))
print(
    "Average actual:",
    round(high["actual_points"].mean(), 3)
)
print(
    "Average predicted:",
    round(high["predicted_points"].mean(), 3)
)
print(
    "Average underprediction:",
    round(
        (
            high["actual_points"] -
            high["predicted_points"]
        ).mean(),
        3
    )
)


# correlation
correlation = results[
    ["actual_points", "predicted_points"]
].corr().iloc[0, 1]

print("\n--- CORRELATION ---")
print(
    "Actual vs predicted:",
    round(correlation, 4)
)


# top-15 ranking
print("\n--- TOP-15 RANKING ---")

overlaps = []

for gw in sorted(results["target_gw"].unique()):

    gw_data = results[
        results["target_gw"] == gw
    ]

    actual_top = set(
        gw_data.nlargest(
            15,
            "actual_points"
        )["element"]
    )

    predicted_top = set(
        gw_data.nlargest(
            15,
            "predicted_points"
        )["element"]
    )

    overlap = len(
        actual_top & predicted_top
    )

    overlaps.append(overlap)

    print(
        f"GW {gw} | "
        f"Top-15 overlap: {overlap}/15"
    )

print(
    "Overall Top-15 recall:",
    round(
        sum(overlaps) /
        (15 * len(overlaps)),
        3
    )
)


# ranking only players who actually played 60+ minutes
if "target_minutes" in results.columns:

    print("\n--- TOP-15 RANKING FOR 60+ MINUTE PLAYERS ---")

    minute_overlaps = []

    for gw in sorted(results["target_gw"].unique()):

        gw_data = results[
            (results["target_gw"] == gw) &
            (results["target_minutes"] >= 60)
        ]

        if len(gw_data) < 15:
            continue

        actual_top = set(
            gw_data.nlargest(
                15,
                "actual_points"
            )["element"]
        )

        predicted_top = set(
            gw_data.nlargest(
                15,
                "predicted_points"
            )["element"]
        )

        overlap = len(
            actual_top & predicted_top
        )

        minute_overlaps.append(overlap)

        print(
            f"GW {gw} | "
            f"Top-15 overlap: {overlap}/15"
        )

    if minute_overlaps:

        print(
            "Overall Top-15 recall (60+):",
            round(
                sum(minute_overlaps) /
                (15 * len(minute_overlaps)),
                3
            )
        )


print(
    "\nPredictions saved to:",
    RESULTS_DIR / "lstm_context_predictions.csv"
)
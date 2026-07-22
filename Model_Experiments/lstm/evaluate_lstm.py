from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# File paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "lstm_prepared"
RESULTS_DIR = BASE_DIR / "results"


# Same model structure used during training
HIDDEN_SIZE = 64
NUM_LAYERS = 1


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing:", device)


# LSTM model
class FPLLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction.squeeze(1)


# Load test data
X_test = np.load(
    DATA_DIR / "X_test.npy"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

metadata = pd.read_csv(
    DATA_DIR / "test_metadata.csv"
)


print("\n--- TEST DATA LOADED ---")

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Metadata:", metadata.shape)


# Convert test data to tensor
X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)


# Create model
input_size = X_test.shape[2]

model = FPLLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(device)


# Load best model found during training
model_path = RESULTS_DIR / "best_lstm_model.pt"

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("\nBest model loaded.")


# Make predictions
with torch.no_grad():

    predictions = model(
        X_test_tensor
    ).cpu().numpy()


# Calculate overall metrics
errors = predictions - y_test

mae = np.mean(
    np.abs(errors)
)

mse = np.mean(
    errors ** 2
)

rmse = np.sqrt(mse)


print("\n--- TEST RESULTS ---")

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


# Save individual predictions
results = metadata.copy()

results["actual_points"] = y_test
results["predicted_points"] = predictions
results["absolute_error"] = np.abs(
    results["predicted_points"]
    - results["actual_points"]
)


prediction_path = (
    RESULTS_DIR
    / "lstm_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)


print(
    "\nPredictions saved to:",
    prediction_path
)


# Performance for different scoring groups
def show_group_metrics(
    label,
    mask
):

    group = results[mask]

    if len(group) == 0:
        return

    group_mae = np.mean(
        np.abs(
            group["predicted_points"]
            - group["actual_points"]
        )
    )

    group_rmse = np.sqrt(
        np.mean(
            (
                group["predicted_points"]
                - group["actual_points"]
            ) ** 2
        )
    )

    print(
        f"{label:<15}"
        f"Count: {len(group):<5} "
        f"MAE: {group_mae:.4f} "
        f"RMSE: {group_rmse:.4f}"
    )


print("\n--- PERFORMANCE BY ACTUAL POINTS ---")

show_group_metrics(
    "0-2 points",
    results["actual_points"].between(
        0,
        2
    )
)

show_group_metrics(
    "3-5 points",
    results["actual_points"].between(
        3,
        5
    )
)

show_group_metrics(
    "6-9 points",
    results["actual_points"].between(
        6,
        9
    )
)

show_group_metrics(
    "10+ points",
    results["actual_points"] >= 10
)


# Show some of the largest mistakes
print("\n--- LARGEST PREDICTION ERRORS ---")

columns_to_show = [
    column
    for column in [
        "name",
        "element",
        "target_gw",
        "actual_points",
        "predicted_points",
        "absolute_error"
    ]
    if column in results.columns
]

print(
    results
    .sort_values(
        "absolute_error",
        ascending=False
    )
    [columns_to_show]
    .head(20)
    .to_string(index=False)
)
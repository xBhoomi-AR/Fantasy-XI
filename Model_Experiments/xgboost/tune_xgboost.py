import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "xgboost_prepared.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load prepared data
df = pd.read_csv(DATA_PATH)

print("\n--- XGBOOST HYPERPARAMETER TUNING ---")
print("Dataset shape:", df.shape)


# Same features used by the final XGBoost model
features = [
    # Previous gameweek
    "total_points_lag1",
    "minutes_lag1",
    "expected_goals_lag1",
    "expected_assists_lag1",
    "expected_goal_involvements_lag1",
    "creativity_lag1",
    "influence_lag1",
    "threat_lag1",
    "ict_index_lag1",

    # Previous 3-game averages
    "total_points_rolling3",
    "minutes_rolling3",
    "expected_goals_rolling3",
    "expected_assists_rolling3",
    "expected_goal_involvements_rolling3",
    "ict_index_rolling3",

    # Previous 5-game averages
    "total_points_rolling5",
    "minutes_rolling5",
    "expected_goals_rolling5",
    "expected_assists_rolling5",
    "expected_goal_involvements_rolling5",
    "ict_index_rolling5",
]

target = "target_points"


# Use only train and validation data for tuning.
# GW 34-38 remains untouched as the final test set.
train_df = df[df["GW"] <= 29].copy()

val_df = df[
    (df["GW"] >= 30) &
    (df["GW"] <= 33)
].copy()

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]


print(
    "Training data:",
    X_train.shape,
    "| GW",
    train_df["GW"].min(),
    "-",
    train_df["GW"].max()
)

print(
    "Validation data:",
    X_val.shape,
    "| GW",
    val_df["GW"].min(),
    "-",
    val_df["GW"].max()
)

print("\nTest set GW 34-38 is not used during tuning.")


# Configurations to compare on the validation set
parameter_configs = [
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
]


results = []

# Train each configuration and compare validation performance
for config_number, params in enumerate(
    parameter_configs,
    start=1
):
    print("\n" + "=" * 60)
    print(f"TRAINING CONFIGURATION {config_number}")
    print("=" * 60)

    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params
    )

    model.fit(
        X_train,
        y_train
    )

    val_predictions = model.predict(
        X_val
    )

    mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_predictions
        )
    )

    r2 = r2_score(
        y_val,
        val_predictions
    )

    print("\nValidation Results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    results.append({
        "config": config_number,
        **params,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2
    })


# Rank configurations using validation MAE
results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "val_mae"
).reset_index(drop=True)

print("\n" + "=" * 70)
print("XGBOOST TUNING RESULTS - SORTED BY VALIDATION MAE")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# Best configuration is the one with the lowest validation MAE
best_result = results_df.iloc[0]

best_config_number = int(
    best_result["config"]
)

best_params = parameter_configs[
    best_config_number - 1
]

print("\n" + "=" * 70)
print("BEST XGBOOST CONFIGURATION")
print("=" * 70)

print(
    "Configuration:",
    best_config_number
)

for key, value in best_params.items():
    print(f"{key}: {value}")

print("\nBEST VALIDATION PERFORMANCE")
print(f"MAE:  {best_result['val_mae']:.4f}")
print(f"RMSE: {best_result['val_rmse']:.4f}")
print(f"R²:   {best_result['val_r2']:.4f}")


# Save all tuning results for reference
output_path = (
    RESULTS_DIR
    / "xgboost_tuning_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nTuning results saved to:")
print(output_path)

print("\n--- TUNING COMPLETE ---")
print(
    "Use the best configuration "
    "for the final XGBoost model."
)
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

print("\n--- PREPARED DATASET ---")
print("Shape:", df.shape)
print("GW range:", df["GW"].min(), "-", df["GW"].max())


# Historical features used by the model
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


# Chronological split to keep future gameweeks out of training
train_df = df[df["GW"] <= 29].copy()

val_df = df[
    (df["GW"] >= 30) &
    (df["GW"] <= 33)
].copy()

test_df = df[df["GW"] >= 34].copy()


X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

X_test = test_df[features]
y_test = test_df[target]


# Verify the split before training
print("\n--- CHRONOLOGICAL DATA SPLIT ---")

print(
    "Train:",
    X_train.shape,
    "| GW",
    train_df["GW"].min(),
    "-",
    train_df["GW"].max()
)

print(
    "Validation:",
    X_val.shape,
    "| GW",
    val_df["GW"].min(),
    "-",
    val_df["GW"].max()
)

print(
    "Test:",
    X_test.shape,
    "| GW",
    test_df["GW"].min(),
    "-",
    test_df["GW"].max()
)

print("\nNumber of model features:", len(features))

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nMissing values:")
print("Train:", X_train.isnull().sum().sum())
print("Validation:", X_val.isnull().sum().sum())
print("Test:", X_test.isnull().sum().sum())


# Final hyperparameters selected using the validation set
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


# Train on the training gameweeks
print("\n--- TRAINING XGBOOST ---")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# Generate validation and test predictions
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)


def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate and print the main regression metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n--- {dataset_name} RESULTS ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return mae, rmse, r2


# Evaluate on validation and test sets
val_mae, val_rmse, val_r2 = evaluate_model(
    y_val,
    val_predictions,
    "VALIDATION"
)

test_mae, test_rmse, test_r2 = evaluate_model(
    y_test,
    test_predictions,
    "TEST"
)


# Show a random sample of test predictions
test_results = test_df[
    ["name", "GW"]
].copy()

test_results["actual_points"] = y_test.values
test_results["predicted_points"] = test_predictions

test_results["absolute_error"] = (
    test_results["actual_points"]
    - test_results["predicted_points"]
).abs()

sample_predictions = test_results.sample(
    n=20,
    random_state=42
).sort_values("GW")

print("\n--- SAMPLE TEST PREDICTIONS ---")
print(sample_predictions.to_string(index=False))


# Save predictions for the complete test set
full_prediction_results = test_df[
    ["name", "GW", "minutes", "target_points"]
].copy()

full_prediction_results = full_prediction_results.rename(
    columns={
        "target_points": "actual_points"
    }
)

full_prediction_results["predicted_points"] = test_predictions

full_prediction_results["absolute_error"] = (
    full_prediction_results["actual_points"]
    - full_prediction_results["predicted_points"]
).abs()

predictions_path = (
    RESULTS_DIR
    / "xgboost_predictions.csv"
)

full_prediction_results.to_csv(
    predictions_path,
    index=False
)

print("\nFull test predictions saved to:")
print(predictions_path)

print(
    "Number of predictions saved:",
    len(full_prediction_results)
)


# Inspect which features the trained model relied on most
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n--- XGBOOST FEATURE IMPORTANCE ---")
print(feature_importance.to_string(index=False))

importance_path = (
    RESULTS_DIR
    / "xgboost_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved to:")
print(importance_path)


# Save the trained model for later use
model_path = (
    RESULTS_DIR
    / "xgboost_model.json"
)

model.save_model(model_path)

print("\nTrained XGBoost model saved to:")
print(model_path)
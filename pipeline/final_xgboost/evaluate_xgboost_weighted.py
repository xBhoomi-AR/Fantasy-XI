from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path("pipeline/final_xgboost/data")
MODEL_DIR = Path("pipeline/final_xgboost/models")
RESULTS_DIR = Path("pipeline/final_xgboost/results")


def load_data():
    print("Loading weighted model...")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    metadata = pd.read_csv("pipeline/final/data/metadata.csv")
    test_metadata = metadata[metadata["season"] == 2526].reset_index(drop=True)

    model = joblib.load(MODEL_DIR / "xgboost_weighted.pkl")

    return X_test, y_test, test_metadata, model


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    corr = np.corrcoef(y_test, predictions)[0, 1]

    print("\nWeighted XGBoost Test Results")
    print(f"MAE         : {mae:.4f}")
    print(f"RMSE        : {rmse:.4f}")
    print(f"Correlation : {corr:.4f}")

    return predictions


def score_ranges(y_true, y_pred):
    print("\nPerformance by score range")

    ranges = [
        ("0-2", 0, 2),
        ("3-5", 3, 5),
        ("6-9", 6, 9),
        ("10+", 10, float("inf")),
    ]

    for name, low, high in ranges:

        if high == float("inf"):
            mask = y_true >= low
        else:
            mask = (y_true >= low) & (y_true <= high)

        if mask.sum() == 0:
            continue

        mae = mean_absolute_error(
            y_true[mask],
            y_pred[mask],
        )

        print(
            f"{name:<5} | Samples: {mask.sum():5d} | MAE: {mae:.4f}"
        )


def save_predictions(metadata, y_true, y_pred):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = metadata.copy()

    results["actual_points"] = y_true
    results["predicted_points"] = y_pred
    results["absolute_error"] = np.abs(y_true - y_pred)

    results.to_csv(
        RESULTS_DIR / "predictions_weighted.csv",
        index=False,
    )

    print("\nWeighted prediction CSV saved.")


def main():

    X_test, y_test, metadata, model = load_data()

    predictions = evaluate(
        model,
        X_test,
        y_test,
    )

    score_ranges(
        y_test,
        predictions,
    )

    save_predictions(
        metadata,
        y_test,
        predictions,
    )


if __name__ == "__main__":
    main()
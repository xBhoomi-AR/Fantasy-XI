import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import torch

from models import BiLSTMModel


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = BASE_DIR / "artifacts" / "experiments"

TEST_X_FILE = ARTIFACT_DIR / "X_test.npy"
TEST_Y_FILE = ARTIFACT_DIR / "y_test.npy"

MODEL_FILE = ARTIFACT_DIR / "bilstm_best.pt"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.pkl"

OUTPUT_FILE = ARTIFACT_DIR / "test_predictions.csv"


# Model settings
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20


def load_data():
    print("Loading test data...")

    X_test = np.load(TEST_X_FILE)
    y_test = np.load(TEST_Y_FILE)

    print(f"Test X shape: {X_test.shape}")
    print(f"Test y shape: {y_test.shape}")

    return X_test, y_test


def scale_features(X_test):
    print("Loading feature scaler...")

    with open(SCALER_FILE, "rb") as file:
        scaler = pickle.load(file)

    n_samples, sequence_length, n_features = X_test.shape

    X_test_2d = X_test.reshape(
        -1,
        n_features
    )

    X_test_scaled = scaler.transform(
        X_test_2d
    )

    X_test_scaled = X_test_scaled.reshape(
        n_samples,
        sequence_length,
        n_features
    ).astype(np.float32)

    print("Test features scaled successfully.")

    return X_test_scaled


def load_model(input_size, device):
    print("Loading trained BiLSTM model...")

    model = BiLSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    model.eval()

    print("Model loaded successfully.")

    return model


def generate_predictions(model, X_test, device):
    print("Generating test predictions...")

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        predictions = model(
            X_tensor
        ).cpu().numpy()

    return predictions


def calculate_metrics(actual, predicted):
    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    return mae, rmse


def analyze_target_ranges(actual, predicted):
    print()
    print("Error by FPL target range")
    print("-------------------------")

    ranges = [
        ("0-2", actual <= 2),
        ("3-5", (actual >= 3) & (actual <= 5)),
        ("6-9", (actual >= 6) & (actual <= 9)),
        ("10+", actual >= 10)
    ]

    results = []

    for name, mask in ranges:

        count = np.sum(mask)

        if count == 0:
            continue

        range_actual = actual[mask]
        range_predicted = predicted[mask]

        mae, rmse = calculate_metrics(
            range_actual,
            range_predicted
        )

        results.append({
            "Target Range": name,
            "Samples": int(count),
            "MAE": mae,
            "RMSE": rmse
        })

        print(
            f"{name:>5} | "
            f"Samples: {count:6d} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f}"
        )

    return results


def main():

    print("Starting BiLSTM evaluation...")
    print()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print()

    X_test, y_test = load_data()

    X_test = scale_features(
        X_test
    )

    model = load_model(
        input_size=X_test.shape[2],
        device=device
    )

    predictions = generate_predictions(
        model,
        X_test,
        device
    )

    actual = y_test

    # Overall metrics
    mae, rmse = calculate_metrics(
        actual,
        predictions
    )

    print()
    print("Overall test results")
    print("--------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Error by target range
    range_results = analyze_target_ranges(
        actual,
        predictions
    )

    # Percentage within one FPL point
    absolute_error = np.abs(
        actual - predictions
    )

    within_one = np.mean(
        absolute_error <= 1.0
    ) * 100

    print()
    print(
        f"Predictions within ±1 FPL point: "
        f"{within_one:.2f}%"
    )

    # Save predictions
    prediction_df = pd.DataFrame({
        "actual_points": actual,
        "predicted_points": predictions,
        "absolute_error": absolute_error
    })

    prediction_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Predictions saved to: {OUTPUT_FILE}"
    )

    # Save range summary
    range_df = pd.DataFrame(
        range_results
    )

    range_file = (
        ARTIFACT_DIR /
        "error_by_target_range.csv"
    )

    range_df.to_csv(
        range_file,
        index=False
    )

    print(
        f"Range analysis saved to: {range_file}"
    )

    print()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
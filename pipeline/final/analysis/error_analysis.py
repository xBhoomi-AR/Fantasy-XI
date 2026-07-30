import os
import numpy as np
import pandas as pd
import torch

from pipeline.final.model.final_model import FinalFPLModel


DATA_DIR = os.path.join("pipeline", "final", "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

MODEL_PATH = os.path.join(
    "pipeline", "final", "saved_models", "final_lstm_best.pt"
)

RESULTS_DIR = os.path.join(
    "pipeline", "final", "results"
)


def load_data():
    print("Loading test data...")

    X_seq = np.load(os.path.join(SPLIT_DIR, "X_seq_test.npy"))
    X_ctx = np.load(os.path.join(SPLIT_DIR, "X_ctx_test.npy"))
    y = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))

    # Need metadata so we can inspect player positions
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

    # Test season is 2025/26
    test_metadata = metadata[
        metadata["season"].astype(str) == "2526"
    ].reset_index(drop=True)

    if len(test_metadata) != len(y):
        raise ValueError(
            f"Metadata/test mismatch: {len(test_metadata)} vs {len(y)}"
        )

    print("Test samples:", len(y))

    return X_seq, X_ctx, y, test_metadata


def load_model(device):
    model = FinalFPLModel(
        sequence_features=32,
        context_features=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.25
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def predict(model, X_seq, X_ctx, device):
    predictions = []
    batch_size = 512

    with torch.no_grad():

        for start in range(0, len(X_seq), batch_size):
            end = start + batch_size

            seq_batch = torch.tensor(
                X_seq[start:end],
                dtype=torch.float32
            ).to(device)

            ctx_batch = torch.tensor(
                X_ctx[start:end],
                dtype=torch.float32
            ).to(device)

            output = model(seq_batch, ctx_batch)

            predictions.append(
                output.cpu().numpy().reshape(-1)
            )

    return np.concatenate(predictions)


def metrics(actual, predicted):
    if len(actual) == 0:
        return np.nan, np.nan, np.nan

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    if (
        len(actual) > 1
        and np.std(actual) > 0
        and np.std(predicted) > 0
    ):
        corr = np.corrcoef(actual, predicted)[0, 1]
    else:
        corr = np.nan

    return mae, rmse, corr


def points_analysis(y, predictions):
    print()
    print("=" * 75)
    print("PERFORMANCE BY ACTUAL FPL POINTS")
    print("=" * 75)

    groups = [
        ("Negative", y < 0),
        ("0-2 points", (y >= 0) & (y <= 2)),
        ("3-5 points", (y >= 3) & (y <= 5)),
        ("6-9 points", (y >= 6) & (y <= 9)),
        ("10+ points", y >= 10),
    ]

    rows = []

    for name, mask in groups:
        actual = y[mask]
        predicted = predictions[mask]

        mae, rmse, corr = metrics(actual, predicted)

        avg_actual = (
            np.mean(actual) if len(actual) else np.nan
        )

        avg_pred = (
            np.mean(predicted) if len(predicted) else np.nan
        )

        rows.append({
            "Group": name,
            "Samples": len(actual),
            "Actual Mean": avg_actual,
            "Predicted Mean": avg_pred,
            "MAE": mae,
            "RMSE": rmse,
            "Correlation": corr
        })

    result = pd.DataFrame(rows)

    print(result.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    return result


def position_analysis(y, predictions, metadata):
    print()
    print("=" * 75)
    print("PERFORMANCE BY POSITION")
    print("=" * 75)

    # Position column name from our final metadata
    if "position" not in metadata.columns:
        print("No position column found in metadata.")
        print("Skipping position analysis.")
        return None

    rows = []

    for position in sorted(metadata["position"].dropna().unique()):

        mask = (
            metadata["position"].astype(str).values
            == str(position)
        )

        actual = y[mask]
        predicted = predictions[mask]

        mae, rmse, corr = metrics(actual, predicted)

        rows.append({
            "Position": position,
            "Samples": len(actual),
            "Actual Mean": np.mean(actual),
            "Predicted Mean": np.mean(predicted),
            "MAE": mae,
            "RMSE": rmse,
            "Correlation": corr
        })

    result = pd.DataFrame(rows)

    print(result.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    return result


def high_score_analysis(y, predictions):
    print()
    print("=" * 75)
    print("HIGH-SCORING PLAYER ANALYSIS")
    print("=" * 75)

    for threshold in [6, 8, 10, 12]:

        mask = y >= threshold

        if mask.sum() == 0:
            continue

        actual = y[mask]
        predicted = predictions[mask]

        print()
        print(f"Actual >= {threshold} points")
        print("Samples:", len(actual))
        print(f"Average actual:    {actual.mean():.2f}")
        print(f"Average predicted: {predicted.mean():.2f}")
        print(
            f"Average underprediction: "
            f"{(actual - predicted).mean():.2f}"
        )


def prediction_distribution(y, predictions):
    print()
    print("=" * 75)
    print("PREDICTION DISTRIBUTION")
    print("=" * 75)

    print(f"Actual mean:       {y.mean():.4f}")
    print(f"Predicted mean:    {predictions.mean():.4f}")

    print(f"Actual std:        {y.std():.4f}")
    print(f"Predicted std:     {predictions.std():.4f}")

    print()
    print(f"Actual maximum:    {y.max():.2f}")
    print(f"Predicted maximum: {predictions.max():.2f}")

    print()
    print(
        "Actual >= 10:",
        int((y >= 10).sum())
    )

    print(
        "Predicted >= 10:",
        int((predictions >= 10).sum())
    )


def save_results(
    y,
    predictions,
    metadata,
    point_results,
    position_results
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = metadata.copy()

    output["actual_points"] = y
    output["predicted_points"] = predictions
    output["error"] = predictions - y
    output["absolute_error"] = np.abs(predictions - y)

    output.to_csv(
        os.path.join(
            RESULTS_DIR,
            "final_test_predictions.csv"
        ),
        index=False
    )

    point_results.to_csv(
        os.path.join(
            RESULTS_DIR,
            "performance_by_points.csv"
        ),
        index=False
    )

    if position_results is not None:
        position_results.to_csv(
            os.path.join(
                RESULTS_DIR,
                "performance_by_position.csv"
            ),
            index=False
        )

    print()
    print("Analysis files saved in:")
    print(RESULTS_DIR)


def main():
    print("=" * 75)
    print("FINAL MODEL ERROR ANALYSIS")
    print("=" * 75)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    X_seq, X_ctx, y, metadata = load_data()

    print("Loading trained model...")
    model = load_model(device)

    print("Generating predictions...")
    predictions = predict(
        model,
        X_seq,
        X_ctx,
        device
    )

    overall_mae, overall_rmse, overall_corr = metrics(
        y,
        predictions
    )

    print()
    print("=" * 75)
    print("OVERALL")
    print("=" * 75)

    print(f"MAE:         {overall_mae:.4f}")
    print(f"RMSE:        {overall_rmse:.4f}")
    print(f"Correlation: {overall_corr:.4f}")

    point_results = points_analysis(
        y,
        predictions
    )

    position_results = position_analysis(
        y,
        predictions,
        metadata
    )

    high_score_analysis(
        y,
        predictions
    )

    prediction_distribution(
        y,
        predictions
    )

    save_results(
        y,
        predictions,
        metadata,
        point_results,
        position_results
    )

    print()
    print("=" * 75)
    print("ERROR ANALYSIS FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    main()
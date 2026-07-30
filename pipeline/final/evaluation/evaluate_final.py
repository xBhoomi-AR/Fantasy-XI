import os
import numpy as np
import torch

from pipeline.final.model.final_model import FinalFPLModel


DATA_DIR = os.path.join("pipeline", "final", "data", "splits")
MODEL_PATH = os.path.join(
    "pipeline", "final", "saved_models", "final_lstm_best.pt"
)


def load_test_data():
    print("Loading test data...")

    X_seq = np.load(os.path.join(DATA_DIR, "X_seq_test.npy"))
    X_context = np.load(os.path.join(DATA_DIR, "X_ctx_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print("Sequence:", X_seq.shape)
    print("Context:", X_context.shape)
    print("Target:", y.shape)

    return X_seq, X_context, y


def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    if np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = np.corrcoef(actual, predicted)[0, 1]
    else:
        correlation = 0.0

    return mae, rmse, correlation


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("FINAL FANTASY-XI MODEL EVALUATION")
    print("=" * 65)
    print("Device:", device)

    X_seq, X_context, y_test = load_test_data()

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

    # Works whether we saved the full training checkpoint
    # or only the model's state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded.")
    print("Running predictions...")

    batch_size = 512
    predictions = []

    with torch.no_grad():

        for start in range(0, len(X_seq), batch_size):
            end = start + batch_size

            seq_batch = torch.tensor(
                X_seq[start:end],
                dtype=torch.float32
            ).to(device)

            context_batch = torch.tensor(
                X_context[start:end],
                dtype=torch.float32
            ).to(device)

            output = model(seq_batch, context_batch)

            predictions.append(
                output.detach().cpu().numpy().reshape(-1)
            )

    predictions = np.concatenate(predictions)

    mae, rmse, correlation = calculate_metrics(
        y_test,
        predictions
    )

    print()
    print("=" * 65)
    print("2025/26 TEST RESULTS")
    print("=" * 65)

    print(f"MAE:         {mae:.4f}")
    print(f"RMSE:        {rmse:.4f}")
    print(f"Correlation: {correlation:.4f}")

    print()
    print("Prediction range:")
    print(
        f"Predicted: {predictions.min():.2f} to "
        f"{predictions.max():.2f}"
    )
    print(
        f"Actual:    {y_test.min():.2f} to "
        f"{y_test.max():.2f}"
    )

    # Simple historical-form baseline:
    # average FPL points from the five previous matches.
    baseline = X_seq[:, :, 0].mean(axis=1)

    baseline_mae, baseline_rmse, baseline_corr = (
        calculate_metrics(y_test, baseline)
    )

    print()
    print("=" * 65)
    print("5-MATCH AVERAGE BASELINE")
    print("=" * 65)

    print(f"MAE:         {baseline_mae:.4f}")
    print(f"RMSE:        {baseline_rmse:.4f}")
    print(f"Correlation: {baseline_corr:.4f}")

    print()
    print("=" * 65)
    print("MODEL VS BASELINE")
    print("=" * 65)

    mae_improvement = (
        (baseline_mae - mae) / baseline_mae
    ) * 100

    rmse_improvement = (
        (baseline_rmse - rmse) / baseline_rmse
    ) * 100

    print(f"MAE improvement:  {mae_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")

    if mae < baseline_mae:
        print("Final model beats baseline on MAE.")
    else:
        print("Baseline beats final model on MAE.")

    if rmse < baseline_rmse:
        print("Final model beats baseline on RMSE.")
    else:
        print("Baseline beats final model on RMSE.")

    print()
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
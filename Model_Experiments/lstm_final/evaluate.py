import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_predictions(model, data_loader, device):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for sequence, context, target in data_loader:
            sequence = sequence.to(device)
            context = context.to(device)

            prediction = model(sequence, context)

            predictions.extend(
                prediction.cpu().numpy()
            )

            targets.extend(
                target.numpy()
            )

    return (
        np.array(predictions),
        np.array(targets)
    )


def calculate_regression_metrics(targets, predictions):
    mae = mean_absolute_error(
        targets,
        predictions
    )

    mse = mean_squared_error(
        targets,
        predictions
    )

    rmse = np.sqrt(mse)

    if len(targets) > 1:
        correlation = np.corrcoef(
            targets,
            predictions
        )[0, 1]
    else:
        correlation = np.nan

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "correlation": float(correlation)
    }


def evaluate_model(model, data_loader, device):
    predictions, targets = get_predictions(
        model,
        data_loader,
        device
    )

    metrics = calculate_regression_metrics(
        targets,
        predictions
    )

    return metrics, predictions, targets
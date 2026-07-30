import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.final.model.final_model import FinalFPLModel


DATA_DIR = Path("pipeline/final/data/splits")
SAVE_DIR = Path("pipeline/final/saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "final_lstm_best.pt"

BATCH_SIZE = 256
MAX_EPOCHS = 100
MIN_EPOCHS = 25

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

EARLY_STOP_PATIENCE = 20


def load_data():
    print("Loading training data...")

    X_seq_train = np.load(DATA_DIR / "X_seq_train.npy")
    X_ctx_train = np.load(DATA_DIR / "X_ctx_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_seq_val = np.load(DATA_DIR / "X_seq_val.npy")
    X_ctx_val = np.load(DATA_DIR / "X_ctx_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    print("Train sequence:", X_seq_train.shape)
    print("Train context: ", X_ctx_train.shape)
    print("Validation sequence:", X_seq_val.shape)
    print("Validation context: ", X_ctx_val.shape)

    train_dataset = TensorDataset(
        torch.tensor(X_seq_train, dtype=torch.float32),
        torch.tensor(X_ctx_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_seq_val, dtype=torch.float32),
        torch.tensor(X_ctx_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequence, context, target in loader:
            sequence = sequence.to(device)
            context = context.to(device)
            target = target.to(device)

            prediction = model(sequence, context)

            loss = criterion(prediction, target)

            total_loss += loss.item() * target.size(0)

            all_predictions.append(prediction.cpu())
            all_targets.append(target.cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    val_loss = total_loss / len(loader.dataset)

    mae = torch.mean(
        torch.abs(predictions - targets)
    ).item()

    mse = torch.mean(
        (predictions - targets) ** 2
    ).item()

    rmse = mse ** 0.5

    if predictions.std() > 0 and targets.std() > 0:
        correlation = torch.corrcoef(
            torch.stack((predictions, targets))
        )[0, 1].item()
    else:
        correlation = 0.0

    return val_loss, mae, rmse, correlation


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("FINAL FANTASY-XI MODEL TRAINING")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available - training on CPU")

    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    train_loader, val_loader = load_data()

    model = FinalFPLModel(
        sequence_features=32,
        context_features=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.25,
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Minimum epochs:", MIN_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print()

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Drop the learning rate if validation gets stuck for a while
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_start = time.perf_counter()
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_train_loss = 0.0

        for sequence, context, target in train_loader:
            sequence = sequence.to(device)
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = model(sequence, context)

            loss = criterion(prediction, target)

            loss.backward()

            # Keeps occasional large LSTM gradients under control
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_train_loss += (
                loss.item() * target.size(0)
            )

        train_loss = (
            total_train_loss / len(train_loader.dataset)
        )

        val_loss, val_mae, val_rmse, val_corr = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"MAE {val_mae:.4f} | "
            f"RMSE {val_rmse:.4f} | "
            f"Corr {val_corr:.4f} | "
            f"LR {current_lr:.6f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_corr": val_corr,
            "learning_rate": current_lr
                })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_corr": val_corr,
                    "sequence_features": 32,
                    "context_features": 14,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.25,
                },
                MODEL_PATH,
            )

            print("  -> New best model saved")

        else:
            epochs_without_improvement += 1

        # Don't allow early stopping during the beginning of training
        if (
            epoch >= MIN_EPOCHS
            and epochs_without_improvement >= EARLY_STOP_PATIENCE
        ):
            print()
            print(
                f"No validation improvement for "
                f"{EARLY_STOP_PATIENCE} epochs."
            )
            print("Stopping training.")
            break

    total_training_time = (
        time.perf_counter() - training_start
    )

    RESULTS_DIR = Path("pipeline/final/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False
    )
    print()
    print("=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Best model:", MODEL_PATH)

    print()
    print(
        "TOTAL TRAINING TIME:",
        format_time(total_training_time)
    )

    print(
        "Finished:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
import numpy as np
import pickle
from pathlib import Path

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models import BiLSTMModel


BASE_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = BASE_DIR / "artifacts" / "experiments"

TRAIN_X_FILE = ARTIFACT_DIR / "X_train.npy"
TRAIN_Y_FILE = ARTIFACT_DIR / "y_train.npy"

VAL_X_FILE = ARTIFACT_DIR / "X_validation.npy"
VAL_Y_FILE = ARTIFACT_DIR / "y_validation.npy"

TEST_X_FILE = ARTIFACT_DIR / "X_test.npy"
TEST_Y_FILE = ARTIFACT_DIR / "y_test.npy"


MODEL_FILE = ARTIFACT_DIR / "bilstm_best.pt"
SCALER_FILE = ARTIFACT_DIR / "feature_scaler.pkl"


BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MAX_EPOCHS = 50
PATIENCE = 10

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.20

GRADIENT_CLIP = 1.0


def load_data():
    print("Loading split datasets...")

    X_train = np.load(TRAIN_X_FILE)
    y_train = np.load(TRAIN_Y_FILE)

    X_val = np.load(VAL_X_FILE)
    y_val = np.load(VAL_Y_FILE)

    X_test = np.load(TEST_X_FILE)
    y_test = np.load(TEST_Y_FILE)

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(X_train, X_val, X_test):
    print("Scaling input features...")

    n_samples, sequence_length, n_features = X_train.shape

    scaler = StandardScaler()

    train_2d = X_train.reshape(
        -1,
        n_features
    )

    scaler.fit(train_2d)

    X_train = scaler.transform(
        train_2d
    ).reshape(
        n_samples,
        sequence_length,
        n_features
    ).astype(np.float32)

    X_val = scaler.transform(
        X_val.reshape(-1, n_features)
    ).reshape(
        X_val.shape
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(
        X_test.shape
    ).astype(np.float32)

    with open(SCALER_FILE, "wb") as file:
        pickle.dump(scaler, file)

    print("Feature scaling completed.")

    return X_train, X_val, X_test


def get_sample_weights(y):
    weights = np.ones(
        len(y),
        dtype=np.float32
    )

    weights[(y >= 3) & (y <= 5)] = 1.5
    weights[(y >= 6) & (y <= 9)] = 3.0
    weights[y >= 10] = 5.0

    return weights


def create_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    train_weights = get_sample_weights(
        y_train
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(train_weights)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def weighted_huber_loss(
    predictions,
    targets,
    weights
):

    loss_function = nn.HuberLoss(
        delta=1.0,
        reduction="none"
    )

    losses = loss_function(
        predictions,
        targets
    )

    weighted_loss = (
        losses * weights
    ).mean()

    return weighted_loss


def calculate_metrics(
    predictions,
    targets
):

    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )

    mae = np.mean(
        np.abs(predictions - targets)
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - targets) ** 2
        )
    )

    return mae, rmse


def train_one_epoch(
    model,
    loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for X_batch, y_batch, weights in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = weighted_huber_loss(
            predictions,
            y_batch,
            weights
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        optimizer.step()

        total_loss += (
            loss.item() *
            X_batch.size(0)
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            y_batch.cpu().numpy()
        )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    mae, rmse = calculate_metrics(
        all_predictions,
        all_targets
    )

    return average_loss, mae, rmse


def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch in loader:

            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            predictions = model(
                X_batch
            )

            loss = nn.HuberLoss()(
                predictions,
                y_batch
            )

            total_loss += (
                loss.item() *
                X_batch.size(0)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y_batch.cpu().numpy()
            )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    mae, rmse = calculate_metrics(
        all_predictions,
        all_targets
    )

    return average_loss, mae, rmse


def main():
    start_time = time.time()

    print("Starting BiLSTM training...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    X_train, X_val, X_test = scale_features(
        X_train,
        X_val,
        X_test
    )

    train_loader, val_loader, test_loader = create_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    model = BiLSTMModel(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print()
    print("Training started...")
    print()

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_mae, train_rmse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss, val_mae, val_rmse = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train MAE: {train_mae:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_FILE
            )

            print(
                "  Best model saved."
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                "Early stopping triggered."
            )

            break

    print()
    print("Loading best model...")

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    test_loss, test_mae, test_rmse = evaluate(
        model,
        test_loader,
        device
    )

    print()
    print("Final test results")
    print("------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    print()
    print(
        f"Best model saved to: {MODEL_FILE}"
    )

    print(
        f"Scaler saved to: {SCALER_FILE}"
    )

    print()
    print("BiLSTM training completed successfully.")

    total_time = time.time() - start_time

    print(f"Total training time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "lstm_prepared"
RESULTS_DIR = BASE_DIR / "results"


# Training settings
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10


# Same random numbers every run
torch.manual_seed(42)
np.random.seed(42)


# Use GPU if available
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
    
    # Load prepared data
X_train = np.load(
    DATA_DIR / "X_train.npy"
)

y_train = np.load(
    DATA_DIR / "y_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

y_val = np.load(
    DATA_DIR / "y_val.npy"
)


print("\n--- DATA LOADED ---")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

# Convert NumPy arrays into tensors
X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)

# Create datasets
train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

# Create data loaders
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

# Create the model
model = FPLLSTM(
    input_size=X_train.shape[2],
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(device)


# Loss function
def weighted_mse_loss(predictions, targets):

    squared_errors = (predictions - targets) ** 2

    weights = torch.ones_like(targets)

    weights = torch.where(
        (targets >= 3) & (targets <= 5),
        1.5,
        weights
    )

    weights = torch.where(
        (targets >= 6) & (targets <= 9),
        2.5,
        weights
    )

    weights = torch.where(
        targets >= 10,
        4.0,
        weights
    )

    return (weights * squared_errors).mean()


criterion = weighted_mse_loss


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n--- MODEL READY ---")
print(model)
print("Number of input features:", X_train.shape[2])
print("Hidden size:", HIDDEN_SIZE)
print("Number of LSTM layers:", NUM_LAYERS)


# Keep track of the best validation loss
best_val_loss = float("inf")
epochs_without_improvement = 0

# Store losses so we can inspect them later
train_losses = []
val_losses = []

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

best_model_path = RESULTS_DIR / "best_lstm_weighted_model.pt"


print("\n--- TRAINING STARTED ---")

for epoch in range(MAX_EPOCHS):

    # Training
    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_dataset)


    # Validation
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_dataset)


    train_losses.append(train_loss)
    val_losses.append(val_loss)


    print(
        f"Epoch {epoch + 1:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    # Save the model when validation improves
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

    else:

        epochs_without_improvement += 1


    # Stop if validation has not improved for a while
    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping after {epoch + 1} epochs"
        )

        break


print("\n--- TRAINING FINISHED ---")
print("Best validation loss:", round(best_val_loss, 4))
print("Best model saved to:", best_model_path)
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
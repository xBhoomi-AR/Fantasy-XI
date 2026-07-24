import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "lstm_context_played_prepared"
RESULTS_DIR = BASE_DIR / "results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# load data
X_seq_train = np.load(DATA_DIR / "X_seq_train.npy")
X_context_train = np.load(DATA_DIR / "X_context_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_seq_val = np.load(DATA_DIR / "X_seq_val.npy")
X_context_val = np.load(DATA_DIR / "X_context_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")


train_data = TensorDataset(
    torch.tensor(X_seq_train, dtype=torch.float32),
    torch.tensor(X_context_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

val_data = TensorDataset(
    torch.tensor(X_seq_val, dtype=torch.float32),
    torch.tensor(X_context_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)


class ContextLSTM(nn.Module):
    def __init__(self, history_size, context_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=history_size,
            hidden_size=64,
            batch_first=True
        )

        self.fc1 = nn.Linear(64 + context_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, sequence, context):
        output, _ = self.lstm(sequence)

        history = output[:, -1, :]

        combined = torch.cat(
            (history, context),
            dim=1
        )

        combined = torch.relu(self.fc1(combined))

        return self.fc2(combined).squeeze(1)


model = ContextLSTM(
    X_seq_train.shape[2],
    X_context_train.shape[1]
).to(device)

print(model)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


best_val_loss = float("inf")
patience = 10
bad_epochs = 0

for epoch in range(1, 101):

    model.train()
    train_loss = 0

    for sequence, context, target in train_loader:

        sequence = sequence.to(device)
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        prediction = model(sequence, context)

        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    model.eval()
    val_loss = 0

    with torch.no_grad():

        for sequence, context, target in val_loader:

            sequence = sequence.to(device)
            context = context.to(device)
            target = target.to(device)

            prediction = model(sequence, context)

            loss = criterion(prediction, target)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            model.state_dict(),
            RESULTS_DIR / "best_lstm_context_played_model.pt"
        )

    else:

        bad_epochs += 1

        if bad_epochs >= patience:
            print("\nEarly stopping")
            break


print("\nTraining finished")
print("Best validation loss:", round(best_val_loss, 4))
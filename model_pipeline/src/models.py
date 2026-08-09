import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):

    def __init__(
        self,
        input_size=56,
        hidden_size=64,
        num_layers=2,
        dropout=0.20
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc1 = nn.Linear(
            hidden_size * 2,
            32
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            32,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        x = self.fc1(last_output)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x.squeeze(1)
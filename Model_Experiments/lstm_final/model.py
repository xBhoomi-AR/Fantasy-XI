import torch
import torch.nn as nn


class ContextLSTM(nn.Module):
    def __init__(
        self,
        sequence_input_size,
        context_input_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        fc_hidden_size=32
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc1 = nn.Linear(
            hidden_size + context_input_size,
            fc_hidden_size
        )

        self.fc2 = nn.Linear(
            fc_hidden_size,
            1
        )

        self.relu = nn.ReLU()

    def forward(self, sequence, context):
        lstm_output, _ = self.lstm(sequence)

        history = lstm_output[:, -1, :]

        combined = torch.cat(
            (history, context),
            dim=1
        )

        x = self.fc1(combined)
        x = self.relu(x)
        prediction = self.fc2(x)

        return prediction.squeeze(1)
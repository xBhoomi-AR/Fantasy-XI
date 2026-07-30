import torch
import torch.nn as nn


class FinalFPLModel(nn.Module):
    def __init__(
        self,
        sequence_features=32,
        context_features=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.25
    ):
        super().__init__()

        # Handles the player's recent match history
        self.lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Handles things known about the upcoming match
        self.context_net = nn.Sequential(
            nn.Linear(context_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combine recent form with the upcoming match context
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, sequence, context):
        lstm_output, _ = self.lstm(sequence)

        # Last timestep summarizes the five previous matches
        history = lstm_output[:, -1, :]

        context = self.context_net(context)

        combined = torch.cat(
            (history, context),
            dim=1
        )

        prediction = self.regressor(combined)

        return prediction.squeeze(1)


def main():
    model = FinalFPLModel()

    # Quick check before we start actual training
    sequence = torch.randn(64, 5, 32)
    context = torch.randn(64, 14)

    output = model(sequence, context)

    print(model)
    print()
    print("Sequence:", sequence.shape)
    print("Context:", context.shape)
    print("Prediction:", output.shape)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", f"{params:,}")


if __name__ == "__main__":
    main()
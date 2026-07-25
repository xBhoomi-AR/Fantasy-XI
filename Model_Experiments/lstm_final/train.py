from pathlib import Path

import torch
import torch.nn as nn


def get_loss_function(loss_name):
    if loss_name == "mse":
        return nn.MSELoss()

    if loss_name == "huber":
        return nn.HuberLoss()

    raise ValueError(f"Unknown loss function: {loss_name}")


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for sequence, context, target in data_loader:
        sequence = sequence.to(device)
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        prediction = model(sequence, context)
        loss = criterion(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for sequence, context, target in data_loader:
            sequence = sequence.to(device)
            context = context.to(device)
            target = target.to(device)

            prediction = model(sequence, context)
            loss = criterion(prediction, target)

            total_loss += loss.item()

    return total_loss / len(data_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device,
    checkpoint_path
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = get_loss_function(config.loss_function)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    best_val_loss = float("inf")
    bad_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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
                checkpoint_path
            )

        else:
            bad_epochs += 1

            if bad_epochs >= config.patience:
                print("Early stopping")
                break

    return history, best_val_loss
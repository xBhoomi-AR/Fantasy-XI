from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Model_Experiments.lstm_final.model import ContextLSTM
from Model_Experiments.lstm_final.train import train_model
from Model_Experiments.lstm_final.evaluate import evaluate_model
from Model_Experiments.lstm_final.utils import (
    get_device,
    log_experiment,
    set_seed
)


def run_experiment(
    train_dataset,
    val_dataset,
    test_dataset,
    config,
    results_dir
):
    # Make the run reproducible
    set_seed(config.seed)

    device = get_device()
    print(f"Using device: {device}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Get feature counts from the data
    sequence_input_size = train_dataset.sequences.shape[2]
    context_input_size = train_dataset.contexts.shape[1]

    model = ContextLSTM(
        sequence_input_size=sequence_input_size,
        context_input_size=context_input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        fc_hidden_size=config.fc_hidden_size
    ).to(device)

    checkpoint_path = (
        results_dir
        / f"{config.experiment_name}_best.pt"
    )

    # Train the model
    history, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path
    )

    # Load the best checkpoint before testing
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )

    # Evaluate on the test set
    metrics, predictions, targets = evaluate_model(
        model,
        test_loader,
        device
    )

    metrics["best_val_loss"] = float(best_val_loss)

    # Save the experiment result
    log_experiment(
        config=config,
        metrics=metrics,
        results_file=results_dir / "experiments.csv"
    )

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
        "checkpoint_path": checkpoint_path
    }
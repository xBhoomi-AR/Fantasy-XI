import shutil
from pathlib import Path

import numpy as np

from Model_Experiments.lstm_final.config import ExperimentConfig
from Model_Experiments.lstm_final.dataset import FPLSequenceDataset
from Model_Experiments.lstm_final.run_experiment import run_experiment


# Fake data only for testing the pipeline
def make_dataset(num_samples):
    sequences = np.random.randn(
        num_samples,
        5,
        27
    )

    contexts = np.random.randn(
        num_samples,
        13
    )

    targets = np.random.randn(
        num_samples
    )

    return FPLSequenceDataset(
        sequences,
        contexts,
        targets
    )


config = ExperimentConfig(
    experiment_name="smoke_test",
    max_epochs=2,
    patience=2,
    batch_size=16
)

train_dataset = make_dataset(100)
val_dataset = make_dataset(40)
test_dataset = make_dataset(40)

results_dir = Path(
    "Model_Experiments/results/lstm_final/smoke_test"
)

result = run_experiment(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    config=config,
    results_dir=results_dir
)

print()
print("Smoke test finished")
print("Metrics:", result["metrics"])
print("Predictions:", result["predictions"].shape)
print("Targets:", result["targets"].shape)
print("Checkpoint:", result["checkpoint_path"])

# Remove the fake results
shutil.rmtree(results_dir)

print("Temporary test results deleted")
import csv
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def log_experiment(config, metrics, results_file):
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    config_data = asdict(config)

    row = {
        **config_data,
        **metrics
    }

    # Store feature lists cleanly inside the CSV
    row["sequence_features"] = ",".join(config.sequence_features)
    row["context_features"] = ",".join(config.context_features)

    file_exists = results_file.exists()

    with results_file.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
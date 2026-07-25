import numpy as np
import torch
from torch.utils.data import Dataset


class FPLSequenceDataset(Dataset):
    def __init__(self, sequences, contexts, targets):
        self.sequences = torch.as_tensor(
            sequences,
            dtype=torch.float32
        )

        self.contexts = torch.as_tensor(
            contexts,
            dtype=torch.float32
        )

        self.targets = torch.as_tensor(
            targets,
            dtype=torch.float32
        )

        if not (
            len(self.sequences)
            == len(self.contexts)
            == len(self.targets)
        ):
            raise ValueError(
                "Sequences, contexts, and targets must have the same number of samples."
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.contexts[index],
            self.targets[index]
        )
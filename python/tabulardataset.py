from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------
# Supervised Dataset for binary classification
# ----------------------------

class TabularDataset(Dataset):
    """
    Thin dataset wrapper for supervised tabular binary classification.

    Expects:
        - X: NumPy array of shape [N, D], float32
        - y: NumPy array of shape [N], float32 with labels in {0.0, 1.0}
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "Features and targets must have the same number of rows."
        self.X = torch.from_numpy(X)  # [N, D], float32
        self.y = torch.from_numpy(y)  # [N], float32

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Returning X, y; y will be unsqueezed in the training loop if needed
        return self.X[idx], self.y[idx]

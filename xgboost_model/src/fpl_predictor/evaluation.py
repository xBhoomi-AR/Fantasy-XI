from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


RANGES = [
    ("0-2", -np.inf, 2),
    ("3-5", 3, 5),
    ("6-9", 6, 9),
    ("10+", 10, np.inf),
]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_frame(y_true, y_pred, positions=None) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = [
        {
            "slice": "overall",
            "n": len(y_true),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "within_1_pct": float(np.mean(np.abs(y_true - y_pred) <= 1) * 100),
            "actual_mean": float(np.mean(y_true)),
            "pred_mean": float(np.mean(y_pred)),
            "actual_ge6_pct": float(np.mean(y_true >= 6) * 100),
            "pred_ge6_pct": float(np.mean(y_pred >= 6) * 100),
        }
    ]
    for name, lo, hi in RANGES:
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.any():
            rows.append(
                {
                    "slice": name,
                    "n": int(mask.sum()),
                    "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
                    "rmse": rmse(y_true[mask], y_pred[mask]),
                    "within_1_pct": float(np.mean(np.abs(y_true[mask] - y_pred[mask]) <= 1) * 100),
                    "actual_mean": float(np.mean(y_true[mask])),
                    "pred_mean": float(np.mean(y_pred[mask])),
                    "actual_ge6_pct": float(np.mean(y_true[mask] >= 6) * 100),
                    "pred_ge6_pct": float(np.mean(y_pred[mask] >= 6) * 100),
                }
            )
    if positions is not None:
        positions = np.asarray(positions)
        for pos in sorted(pd.Series(positions).dropna().unique()):
            mask = positions == pos
            if mask.any():
                rows.append(
                    {
                        "slice": f"position_{pos}",
                        "n": int(mask.sum()),
                        "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
                        "rmse": rmse(y_true[mask], y_pred[mask]),
                        "within_1_pct": float(np.mean(np.abs(y_true[mask] - y_pred[mask]) <= 1) * 100),
                        "actual_mean": float(np.mean(y_true[mask])),
                        "pred_mean": float(np.mean(y_pred[mask])),
                        "actual_ge6_pct": float(np.mean(y_true[mask] >= 6) * 100),
                        "pred_ge6_pct": float(np.mean(y_pred[mask] >= 6) * 100),
                    }
                )
    return pd.DataFrame(rows)


def range_weights(y: pd.Series, high_multiplier: float = 1.0) -> np.ndarray:
    bins = pd.cut(y, bins=[-np.inf, 2, 5, 9, np.inf], labels=["0-2", "3-5", "6-9", "10+"])
    counts = bins.value_counts()
    weights = bins.map(lambda b: np.sqrt(len(y) / max(counts.get(b, 1), 1))).astype(float).to_numpy()
    high_boost = np.select([y >= 10, y >= 6, y >= 3], [2.0, 1.6, 1.15], default=1.0)
    weights = weights * (1 + (high_boost - 1) * high_multiplier)
    return weights / np.mean(weights)

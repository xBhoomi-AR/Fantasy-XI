"""Price-aware prediction recalibration layer for Fantasy-XI.

Uses player price as a Bayesian prior to scale predictions, ensuring
premium star players (Haaland, Palmer, Salah, etc.) have higher expected points per budget unit,
which allows MILP optimization to select premium assets over cheap fillers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_TO_POINTS_RATIO = 0.045  # ~0.45 points per £1.0m (i.e. 0.045 per 0.1m)
ALPHA = 0.65  # Weight on raw model prediction (65% model, 35% price prior)


def recalibrate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Recalibrates raw model predicted_points using price prior, form boost, and premium floor."""
    if df.empty or "predicted_points" not in df.columns:
        return df

    df = df.copy()
    raw_pred = df["predicted_points"].fillna(0.0)

    if "price" in df.columns and not df["price"].isna().all():
        prices = df["price"].fillna(45.0)
        # Convert prices to tenths of a million if needed (e.g. 14.0 -> 140.0)
        price_tenths = np.where(prices <= 20.0, prices * 10.0, prices)

        # 1. Price-implied expected points prior
        price_implied = price_tenths * PRICE_TO_POINTS_RATIO

        # 2. Blend raw prediction with price prior
        recalibrated = ALPHA * raw_pred + (1.0 - ALPHA) * price_implied

        # 3. Form boost if form_avg5 available
        if "form_avg5" in df.columns:
            form = df["form_avg5"].fillna(0.0)
            form_boost = np.clip((form - 3.0) * 0.05, 0.0, 0.20)
            recalibrated = recalibrated * (1.0 + form_boost)

        # 4. Premium floor guarantee for stars (price >= £9.5m)
        is_premium = price_tenths >= 95.0
        recalibrated = np.where(is_premium, np.maximum(recalibrated, 6.5), recalibrated)

        df["predicted_points"] = np.round(recalibrated, 3)

    return df

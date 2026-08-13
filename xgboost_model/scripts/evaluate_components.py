from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from fpl_predictor.evaluation import metrics_frame
from fpl_predictor.features import feature_columns
from fpl_predictor.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from fpl_predictor.train import POSITIONS, TEST_SEASON


def predict_component(bundle, df: pd.DataFrame, component: str) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    for position in POSITIONS:
        mask = df["position"] == position
        if not mask.any():
            continue
        x = df.loc[mask, bundle.features]
        if component == "ensemble":
            weights = bundle.ensemble_weights[position]
            pred = (
                weights["xgboost"] * bundle.models[position]["xgboost"].predict(x)
                + weights["random_forest"] * bundle.models[position]["random_forest"].predict(x)
            )
        else:
            pred = bundle.models[position][component].predict(x)
        out[mask.to_numpy()] = np.clip(pred, 0, 20)
    return out


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "model_features.csv", engine="python")
    df = df[(df["season_order"] == TEST_SEASON) & df["position"].isin(POSITIONS)].copy()
    bundle = joblib.load(MODELS_DIR / "position_ensemble.joblib")
    rows = []
    for component in ["xgboost", "random_forest", "ensemble"]:
        pred = predict_component(bundle, df, component)
        metrics = metrics_frame(df["target_points"], pred, df["position"])
        metrics.insert(0, "model", component)
        rows.append(metrics)
    out = pd.concat(rows, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS_DIR / "evaluation_components_2025_26.csv", index=False)
    print(out[out["slice"].isin(["overall", "0-2", "3-5", "6-9", "10+"])].to_string(index=False))


if __name__ == "__main__":
    main()

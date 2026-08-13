from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / ".codex_pydeps"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fpl_predictor.features import build_features, feature_columns


if __name__ == "__main__":
    df = build_features()
    print(f"Wrote data/processed/model_features.csv with {len(df):,} rows and {len(feature_columns(df))} model features.")

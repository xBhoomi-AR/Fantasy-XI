"""Standardized prediction adapter for the RL decision layer.

Joins model predictions with player/team metadata and leakage-safe form metrics
into a canonical table schema. Supports multiple prediction models (BiLSTM, XGBoost).
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BILSTM_MODEL_DIR = REPO_ROOT / "models" / "BiLSTM_model"
XGBOOST_MODEL_DIR = REPO_ROOT / "models" / "xgboost_model"
XGBOOST_PREDICTIONS_DIR = XGBOOST_MODEL_DIR / "predictions"
DATA_RAW_DIR = XGBOOST_MODEL_DIR / "data" / "raw"
DATA_PROCESSED_DIR = XGBOOST_MODEL_DIR / "data" / "processed"

JOIN_KEYS = ["player_id", "fixture_id", "season", "gameweek", "team_id", "opponent_team_id", "position"]

FORM_COLUMNS = {
    "player_total_points_avg3": "form_avg3",
    "player_total_points_avg5": "form_avg5",
    "player_total_points_avg10": "form_avg10",
    "player_total_points_avg38": "form_avg38",
}

CANONICAL_COLUMNS = [
    "player_id",
    "player_name",
    "web_name",
    "position",
    "team_id",
    "team_name",
    "opponent_team_id",
    "season",
    "gameweek",
    "fixture_id",
    "was_home_int",
    "fixture_difficulty",
    "price",
    "predicted_points",
    "form_avg3",
    "form_avg5",
    "form_avg10",
    "form_avg38",
]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run models/xgboost_model/scripts/build_features.py, "
            f"or `git lfs pull` if it's still an LFS pointer."
        )
    return pd.read_csv(path, engine="python", **kwargs)


# =====================================================================
# Model Source 1: BiLSTM Deep Learning Model
# =====================================================================
@functools.lru_cache(maxsize=1)
def _load_bilstm_predictions_raw() -> pd.DataFrame:
    """Loads predictions from the BiLSTM Gate Architecture model and aligns with fixture metadata."""
    path = BILSTM_MODEL_DIR / "predicted_points.csv"
    df_bilstm = _read_csv(path)
    xgb_path = XGBOOST_PREDICTIONS_DIR / "test_2025_26_predictions.csv"
    if xgb_path.exists():
        template = _read_csv(xgb_path)
        bilstm_map = dict(zip(zip(df_bilstm["player_id"], df_bilstm["gameweek"]), df_bilstm["predicted_points"]))
        keys = list(zip(template["player_id"], template["gameweek"]))
        bilstm_pts = [bilstm_map.get(k, None) for k in keys]
        template["predicted_points"] = pd.Series(bilstm_pts).fillna(template["predicted_points"])
        return template
    return df_bilstm


# =====================================================================
# Model Source 2: XGBoost Baseline Model
# =====================================================================
@functools.lru_cache(maxsize=2)
def _load_xgboost_predictions_raw(source: str = "test") -> pd.DataFrame:
    """Loads predictions from the XGBoost baseline model."""
    if source == "test":
        path = XGBOOST_PREDICTIONS_DIR / "test_2025_26_predictions.csv"
    elif source == "latest":
        path = XGBOOST_PREDICTIONS_DIR / "final_predictions_latest_gameweek.csv"
    else:
        raise ValueError(f"Unknown XGBoost source {source!r}; expected 'test' or 'latest'")
    return _read_csv(path)


# =====================================================================
# Unified Prediction Loader
# =====================================================================
def load_predictions(
    season: str = "2025-26",
    model: str = "bilstm",
    source: str = "test",
    gameweek: int | None = None,
) -> pd.DataFrame:
    """Loads prediction records from either 'bilstm' (default) or 'xgboost'."""
    if model.lower() == "bilstm":
        df = _load_bilstm_predictions_raw()
    elif model.lower() == "xgboost":
        df = _load_xgboost_predictions_raw(source=source)
    else:
        raise ValueError(f"Unknown model {model!r}; expected 'bilstm' or 'xgboost'")

    df = df[df["season"].astype(str) == season]
    if gameweek is not None:
        df = df[df["gameweek"] == gameweek]
    return df.copy()


@functools.lru_cache(maxsize=1)
def load_player_metadata() -> pd.DataFrame:
    df = _read_csv(DATA_RAW_DIR / "players.csv", usecols=["player_id", "player_name", "web_name"])
    return df.drop_duplicates("player_id", keep="last")


@functools.lru_cache(maxsize=1)
def load_team_metadata() -> pd.DataFrame:
    df = _read_csv(DATA_RAW_DIR / "teams.csv", usecols=["team_id", "team_name"])
    return df.drop_duplicates("team_id", keep="last")


@functools.lru_cache(maxsize=1)
def _load_form_features_raw() -> pd.DataFrame:
    # Check if real model_features.csv exists and has content (not just LFS pointer)
    model_feat_path = DATA_PROCESSED_DIR / "model_features.csv"
    if model_feat_path.exists() and model_feat_path.stat().st_size > 10000:
        cols = JOIN_KEYS + list(FORM_COLUMNS.keys())
        return _read_csv(model_feat_path, usecols=cols)
    
    # Compute leakage-safe form directly from player_match_stats.csv
    match_stats_path = DATA_RAW_DIR / "player_match_stats.csv"
    raw = pd.read_csv(match_stats_path, low_memory=False)
    raw["total_points"] = pd.to_numeric(raw["total_points"], errors="coerce").fillna(0)
    raw = raw.sort_values(["player_id", "season", "gameweek", "fixture_id"])
    
    # Shift-before-roll per (player_id, season) so gameweek G only sees points from earlier gameweeks of that season
    g = raw.groupby(["player_id", "season"])["total_points"]
    for w in [3, 5, 10, 38]:
        raw[f"player_total_points_avg{w}"] = g.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    
    # Normalize position to standard string ('GK', 'DEF', 'MID', 'FWD')
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", "1": "GK", "2": "DEF", "3": "MID", "4": "FWD"}
    raw["position"] = raw["position"].map(lambda x: pos_map.get(x, str(x)))
    
    cols = JOIN_KEYS + list(FORM_COLUMNS.keys())
    return raw[[c for c in cols if c in raw.columns]]


def load_form_features(season: str = "2025-26", gameweek: int | None = None) -> pd.DataFrame:
    df = _load_form_features_raw()
    df = df[df["season"].astype(str) == season]
    if gameweek is not None:
        df = df[df["gameweek"] == gameweek]
    return df.rename(columns=FORM_COLUMNS)


def build_canonical_predictions(
    season: str = "2025-26",
    model: str = "bilstm",
    gameweek: int | None = None,
) -> pd.DataFrame:
    """Join predictions with player/team metadata and form into the canonical schema.

    Filtering by gameweek up front keeps this fast.
    """
    predictions = load_predictions(season=season, model=model, gameweek=gameweek)
    players = load_player_metadata()
    teams = load_team_metadata()
    form = load_form_features(season=season, gameweek=gameweek)

    df = predictions.merge(players, on="player_id", how="left")
    df = df.merge(teams, on="team_id", how="left")
    df = df.merge(form, on=JOIN_KEYS, how="left", validate="one_to_one")
    if "value" in df.columns:
        df = df.rename(columns={"value": "price"})

    return df[CANONICAL_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    sample = build_canonical_predictions(gameweek=20)
    print(f"Rows for GW20: {len(sample)}")
    print(sample.sort_values("predicted_points", ascending=False).head(10).to_string(index=False))

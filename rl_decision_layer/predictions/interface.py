"""Builds one standardized prediction table from the existing XGBoost pipeline's output.

Joins predictions/*.csv with player/team metadata and the recent-form columns
already computed in model_features.csv. Doesn't compute anything new itself -
all feature engineering still happens in models/xgboost_model/src/fpl_predictor/features.py.

Source of each field:
  player_id, team_id, opponent_team_id, fixture_id, season, gameweek, position,
  was_home_int, fixture_difficulty, price, predicted_points
      -> models/xgboost_model/predictions/*.csv (price is `value` renamed - it's
         FPL's tenths-of-a-million price, e.g. 53 = 5.3m, confirmed from the
         5-154 range across the whole dataset)
  player_name, web_name  -> data/raw/players.csv, joined on player_id
  team_name               -> data/raw/teams.csv, joined on team_id
  form_avg3/5/10/38       -> data/processed/model_features.csv, joined on
      [player_id, fixture_id, season, gameweek, team_id, opponent_team_id, position]
      (same join keys scripts/ranking_evaluation.py already uses). These are
      shifted before rolling upstream, so form at gameweek G only reflects
      gameweeks before G.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
XGBOOST_MODEL_DIR = REPO_ROOT / "models" / "xgboost_model"
PREDICTIONS_DIR = XGBOOST_MODEL_DIR / "predictions"
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


def load_predictions(season: str = "2025-26", source: str = "test", gameweek: int | None = None) -> pd.DataFrame:
    # "test" covers every gameweek of the 2025-26 test season, which is what we need
    # for historical candidate generation. "latest" is just the final gameweek.
    if source == "test":
        path = PREDICTIONS_DIR / "test_2025_26_predictions.csv"
    elif source == "latest":
        path = PREDICTIONS_DIR / "final_predictions_latest_gameweek.csv"
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'test' or 'latest'")

    df = _read_csv(path)
    df = df[df["season"].astype(str) == season]
    if gameweek is not None:
        df = df[df["gameweek"] == gameweek]
    return df.copy()


def load_player_metadata() -> pd.DataFrame:
    df = _read_csv(DATA_RAW_DIR / "players.csv", usecols=["player_id", "player_name", "web_name"])
    return df.drop_duplicates("player_id", keep="last")


def load_team_metadata() -> pd.DataFrame:
    df = _read_csv(DATA_RAW_DIR / "teams.csv", usecols=["team_id", "team_name"])
    return df.drop_duplicates("team_id", keep="last")


def load_form_features(season: str = "2025-26", gameweek: int | None = None) -> pd.DataFrame:
    cols = JOIN_KEYS + list(FORM_COLUMNS.keys())
    df = _read_csv(DATA_PROCESSED_DIR / "model_features.csv", usecols=cols)
    df = df[df["season"].astype(str) == season]
    if gameweek is not None:
        df = df[df["gameweek"] == gameweek]
    return df.rename(columns=FORM_COLUMNS)


def build_canonical_predictions(
    season: str = "2025-26",
    source: str = "test",
    gameweek: int | None = None,
) -> pd.DataFrame:
    """Join predictions with player/team metadata and form into the canonical schema.

    Filtering by gameweek up front keeps this fast even though
    model_features.csv has ~250k rows.
    """
    predictions = load_predictions(season=season, source=source, gameweek=gameweek)
    players = load_player_metadata()
    teams = load_team_metadata()
    form = load_form_features(season=season, gameweek=gameweek)

    df = predictions.merge(players, on="player_id", how="left")
    df = df.merge(teams, on="team_id", how="left")
    df = df.merge(form, on=JOIN_KEYS, how="left", validate="one_to_one")
    df = df.rename(columns={"value": "price"})

    return df[CANONICAL_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    sample = build_canonical_predictions(gameweek=20)
    print(f"Rows for GW20: {len(sample)}")
    print(sample.sort_values("predicted_points", ascending=False).head(10).to_string(index=False))

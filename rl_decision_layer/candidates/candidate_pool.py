"""Candidate pool for a single target gameweek.

Combines three sources per position, not just the DL top-K:
  - prediction top-K (highest predicted_points)
  - current squad (kept regardless of rank)
  - recent form (strong form_avg5 even if predicted_points rank is lower)

Everything here operates on the canonical table from predictions/interface.py,
which is already leakage-safe (predictions and form for gameweek G only use
data from before G). This module just filters to target_gameweek and ranks -
it doesn't touch any actual points from G.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .ranking import FPL_POSITIONS, rank_by_position

# An FPL squad is 2 GK / 5 DEF / 5 MID / 3 FWD. These defaults give each
# position roughly 3-4x its squad-slot count worth of prediction candidates,
# which should be enough transfer options for MILP without blowing up its
# problem size. Not empirically tuned - override per experiment.
DEFAULT_PREDICTION_TOP_K = {"GK": 8, "DEF": 15, "MID": 15, "FWD": 10}
DEFAULT_FORM_TOP_K = {"GK": 3, "DEF": 5, "MID": 5, "FWD": 4}
DEFAULT_FORM_WINDOW = "form_avg5"


def _per_position(value: int | dict[str, int], positions: Sequence[str]) -> dict[str, int]:
    if isinstance(value, dict):
        return {p: value.get(p, 0) for p in positions}
    return {p: value for p in positions}


def _collapse_double_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """A player with two fixtures in the same gameweek (a DGW) should rank on
    their combined predicted points, not appear as two separate, lower-scoring
    rows. We sum predicted_points across fixtures and keep one row per player,
    recording how many fixtures it covered. Fixture-specific fields
    (opponent, home/away, difficulty) end up reflecting just the first
    fixture - fine for ranking, but something MILP will need to handle
    properly once it deals with DGWs directly.
    """
    fixture_counts = df.groupby("player_id")["fixture_id"].transform("nunique")
    if (fixture_counts <= 1).all():
        return df.assign(fixture_count=1)

    df = df.copy()
    df["fixture_count"] = fixture_counts
    df["predicted_points"] = df.groupby("player_id")["predicted_points"].transform("sum")
    return df.drop_duplicates("player_id", keep="first")


def build_candidate_pool(
    canonical_df: pd.DataFrame,
    target_gameweek: int,
    current_squad_ids: Sequence[int] = (),
    prediction_top_k: int | dict[str, int] = DEFAULT_PREDICTION_TOP_K,
    form_top_k: int | dict[str, int] = DEFAULT_FORM_TOP_K,
    form_window: str = DEFAULT_FORM_WINDOW,
) -> pd.DataFrame:
    """Build the candidate pool for target_gameweek.

    A squad player with no row for target_gameweek (no fixture that week)
    can't be "retained" since there's nothing to retain - they just won't
    appear. Squad legality (2/5/5/3) isn't checked here, that's MILP's job.

    Returns the canonical columns plus `candidate_source` (which of
    current_squad / prediction_topk / recent_form picked this row, comma
    separated) and `fixture_count` (>1 for a collapsed double gameweek).
    """
    gw_df = canonical_df[canonical_df["gameweek"] == target_gameweek].copy()
    if gw_df.empty:
        raise ValueError(f"No rows for target_gameweek={target_gameweek}")

    gw_df = _collapse_double_gameweeks(gw_df)

    pred_k = _per_position(prediction_top_k, FPL_POSITIONS)
    form_k = _per_position(form_top_k, FPL_POSITIONS)
    squad_ids = set(current_squad_ids)

    ranked = rank_by_position(gw_df, top_k=None)

    pool_frames = []
    for position, pos_df in ranked.items():
        source = {}

        for pid in pos_df.head(pred_k[position])["player_id"]:
            source.setdefault(pid, set()).add("prediction_topk")

        for pid in pos_df[pos_df["player_id"].isin(squad_ids)]["player_id"]:
            source.setdefault(pid, set()).add("current_squad")

        remaining = pos_df[~pos_df["player_id"].isin(source.keys())]
        for pid in remaining.sort_values(form_window, ascending=False).head(form_k[position])["player_id"]:
            source.setdefault(pid, set()).add("recent_form")

        selected = pos_df[pos_df["player_id"].isin(source.keys())].copy()
        selected["candidate_source"] = selected["player_id"].map(lambda pid: ",".join(sorted(source[pid])))
        pool_frames.append(selected)

    pool = pd.concat(pool_frames, ignore_index=True)
    return pool.sort_values(["position", "predicted_points"], ascending=[True, False]).reset_index(drop=True)

"""Position-wise ranking. GK/DEF/MID/FWD are never comparable on one scale,
so we always rank within a position, never across all positions at once."""

from __future__ import annotations

import pandas as pd

FPL_POSITIONS = ["GK", "DEF", "MID", "FWD"]


def rank_by_position(
    df: pd.DataFrame,
    top_k: int | dict[str, int] | None = None,
    score_col: str = "predicted_points",
) -> dict[str, pd.DataFrame]:
    """Rank each position separately by score_col, descending.

    top_k can be a single int applied to every position, a dict for
    per-position cutoffs (squad slots differ: 2 GK / 5 DEF / 5 MID / 3 FWD),
    or None to keep everyone.
    """
    ranked = {}
    for position in FPL_POSITIONS:
        pos_df = df[df["position"] == position]
        if pos_df.empty:
            continue
        pos_df = pos_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        pos_df.insert(0, "rank", pos_df.index + 1)

        k = top_k.get(position) if isinstance(top_k, dict) else top_k
        if k is not None:
            pos_df = pos_df.head(k)

        ranked[position] = pos_df

    return ranked

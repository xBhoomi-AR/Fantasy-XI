"""Rule-based chip selection heuristics for Fantasy-XI.

Overrides or guides PPO's chip actions based on game state conditions:
- Triple Captain: High predicted score for active captain (>= 8.5)
- Bench Boost: High predicted score for bench players (>= 12.0 combined)
- Free Hit: High fixture difficulty across starting XI or blank GW
- Wildcard: When squad transfers needed >= 5 or after GW 19
"""

from __future__ import annotations

import pandas as pd


def get_recommended_chip(
    gameweek: int,
    squad_rows: pd.DataFrame,
    xi_starting_ids: list[int],
    xi_bench_ids: list[int],
    captain_id: int | None,
    available_chips: dict[str, bool],
) -> str:
    """Returns 'triple_captain', 'bench_boost', 'free_hit', 'wildcard', or 'none'."""

    # 1. Triple Captain Heuristic
    if available_chips.get("triple_captain", False) and captain_id is not None:
        cap_row = squad_rows[squad_rows["player_id"] == captain_id]
        if not cap_row.empty:
            cap_pred = cap_row.iloc[0].get("predicted_points", 0.0)
            if cap_pred >= 8.5:
                return "triple_captain"

    # 2. Bench Boost Heuristic
    if available_chips.get("bench_boost", False):
        bench_rows = squad_rows[squad_rows["player_id"].isin(xi_bench_ids)]
        bench_pred_sum = bench_rows["predicted_points"].sum() if not bench_rows.empty else 0.0
        if bench_pred_sum >= 14.0:
            return "bench_boost"

    # 3. Free Hit Heuristic (Blank / Tough Fixtures)
    if available_chips.get("free_hit", False):
        starters = squad_rows[squad_rows["player_id"].isin(xi_starting_ids)]
        if not starters.empty and "fixture_difficulty" in starters.columns:
            hard_fixtures = (starters["fixture_difficulty"] >= 4.5).sum()
            if hard_fixtures >= 4:
                return "free_hit"

    # 4. Wildcard Heuristic (Mid-season reset around GW 19-24)
    if available_chips.get("wildcard", False) and gameweek in (19, 20, 21, 22):
        return "wildcard"

    return "none"

"""Dynamic, opportunity-aware chip selection heuristics for Fantasy-XI.

Evaluates game-state opportunities (Double Gameweeks, Blank Gameweeks, High Captain Ceilings)
to deploy chips at maximum expected point yield:
- Triple Captain: Double Gameweek for Captain or predicted points >= 10.0
- Bench Boost: Double Gameweek for Bench players or bench expected points >= 15.0
- Free Hit: Blank Gameweek (players missing fixtures) or severe fixture difficulty cluster
- Wildcard: Mid-season fixture swing (GW 19-28) when squad needs major restructuring
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

    # 1. Triple Captain (DGW or High Prediction)
    if available_chips.get("triple_captain", False) and captain_id is not None:
        cap_row = squad_rows[squad_rows["player_id"] == captain_id]
        if not cap_row.empty:
            cap_pred = cap_row.iloc[0].get("predicted_points", 0.0)
            is_dgw = cap_row.iloc[0].get("fixture_count", 1) > 1
            if is_dgw or cap_pred >= 10.0:
                return "triple_captain"

    # 2. Bench Boost (DGW or High Bench Expectation)
    if available_chips.get("bench_boost", False):
        bench_rows = squad_rows[squad_rows["player_id"].isin(xi_bench_ids)]
        if not bench_rows.empty:
            bench_pred_sum = bench_rows["predicted_points"].sum()
            has_bench_dgw = "fixture_count" in bench_rows.columns and (bench_rows["fixture_count"] > 1).sum() >= 2
            if has_bench_dgw or bench_pred_sum >= 15.0:
                return "bench_boost"

    # 3. Free Hit (Blank GW or Fixture Difficulty Cluster)
    if available_chips.get("free_hit", False):
        starters = squad_rows[squad_rows["player_id"].isin(xi_starting_ids)]
        if not starters.empty and "fixture_difficulty" in starters.columns:
            hard_fixtures = (starters["fixture_difficulty"] >= 4.5).sum()
            blank_fixtures = (starters["fixture_difficulty"].isna()).sum()
            if hard_fixtures >= 4 or blank_fixtures >= 3:
                return "free_hit"

    # 4. Wildcard (Mid-season fixture swing around GW 19-24)
    if available_chips.get("wildcard", False) and gameweek in (19, 20, 21, 24):
        return "wildcard"

    return "none"

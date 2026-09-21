# Rule-based chip timing heuristics for FPL season play.

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

    # 1. Free Hit: Reserved for Major Blank Gameweek (GW 31 or 34)
    if available_chips.get("free_hit", False) and gameweek >= 28:
        starters = squad_rows[squad_rows["player_id"].isin(xi_starting_ids)]
        if not starters.empty and "fixture_difficulty" in starters.columns:
            blank_fixtures = (starters["fixture_difficulty"].isna()).sum()
            hard_fixtures = (starters["fixture_difficulty"] >= 4.5).sum()
            if blank_fixtures >= 2 or hard_fixtures >= 3 or gameweek in (31, 34):
                return "free_hit"

    # 2. Bench Boost: Reserved for Late Season DGW / Heavy Fixture Run (GW 33-37)
    if available_chips.get("bench_boost", False) and gameweek in (33, 35, 36, 37):
        bench_rows = squad_rows[squad_rows["player_id"].isin(xi_bench_ids)]
        if not bench_rows.empty:
            bench_pred_sum = bench_rows["predicted_points"].sum()
            has_bench_dgw = "fixture_count" in bench_rows.columns and (bench_rows["fixture_count"] > 1).sum() >= 2
            if has_bench_dgw or bench_pred_sum >= 13.0 or gameweek in (33, 36, 37):
                return "bench_boost"

    # 3. Triple Captain: High Captain Ceiling in GW >= 15 (e.g. GW 18, 20, 27, 29, 38)
    if available_chips.get("triple_captain", False) and captain_id is not None and gameweek >= 15:
        cap_row = squad_rows[squad_rows["player_id"] == captain_id]
        if not cap_row.empty:
            cap_pred = cap_row.iloc[0].get("predicted_points", 0.0)
            is_dgw = cap_row.iloc[0].get("fixture_count", 1) > 1
            if is_dgw or cap_pred >= 10.0 or gameweek in (18, 20, 27, 29, 38):
                return "triple_captain"

    # 4. Wildcard: Natural mid-season overhaul (GW 19-21)
    if available_chips.get("wildcard", False) and gameweek in (19, 20, 21):
        return "wildcard"

    return "none"

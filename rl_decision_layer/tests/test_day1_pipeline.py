"""Smoke tests for the candidate-generation foundation.

Run with:
    python rl_decision_layer/tests/test_day1_pipeline.py

Needs models/xgboost_model/data/processed/model_features.csv to have real
content, not a Git LFS pointer. If it's a pointer, run
models/xgboost_model/scripts/build_features.py to regenerate it locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from rl_decision_layer.candidates.candidate_pool import build_candidate_pool
from rl_decision_layer.candidates.pipeline import get_candidates
from rl_decision_layer.candidates.ranking import FPL_POSITIONS, rank_by_position
from rl_decision_layer.predictions.interface import DATA_RAW_DIR, build_canonical_predictions

# Spread across the season rather than one arbitrary gameweek.
TEST_GAMEWEEKS = [5, 15, 35]
DGW_GAMEWEEK = 26  # a real double gameweek in the 2025-26 test data

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def build_test_squad(df: pd.DataFrame) -> list[int]:
    """Pick a 15-man squad from the middle of the pack (not top predicted),
    so retention tests actually prove something."""
    squad = []
    for position, n in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
        players = df[df["position"] == position].sort_values("predicted_points")
        start = max(0, len(players) // 2 - n // 2)
        squad.extend(players["player_id"].iloc[start:start + n].tolist())
    return squad


def test_schema_and_ranking(gw: int) -> pd.DataFrame:
    df = build_canonical_predictions(gameweek=gw)
    check(len(df) > 0, f"GW{gw}: canonical predictions non-empty ({len(df)} rows)")
    check(df["player_id"].notna().all(), f"GW{gw}: no null player_id")
    check(df["player_name"].notna().mean() > 0.95, f"GW{gw}: player_name resolved for >95%")
    check(df["team_name"].notna().mean() > 0.95, f"GW{gw}: team_name resolved for >95%")
    check(set(df["position"].unique()) <= set(FPL_POSITIONS), f"GW{gw}: positions are all valid")

    ranked = rank_by_position(df)
    for position, pos_df in ranked.items():
        sorted_desc = (pos_df["predicted_points"].diff().dropna() <= 0).all()
        check(sorted_desc, f"GW{gw} {position}: ranked descending by predicted_points")
        check((pos_df["position"] == position).all(), f"GW{gw} {position}: ranked group is single-position")

    return df


def test_candidate_pool(gw: int, df: pd.DataFrame) -> None:
    from rl_decision_layer.candidates.candidate_pool import DEFAULT_FORM_TOP_K, DEFAULT_PREDICTION_TOP_K

    squad = build_test_squad(df)
    pool = build_candidate_pool(df, target_gameweek=gw, current_squad_ids=squad)
    check(len(pool) > 0, f"GW{gw}: candidate pool non-empty ({len(pool)} rows)")

    squad_in_gw = set(squad) & set(df["player_id"])
    missing = squad_in_gw - set(pool["player_id"])
    check(len(missing) == 0, f"GW{gw}: all squad players with a fixture that week are retained")

    squad_rows = pool[pool["player_id"].isin(squad)]
    check(squad_rows["candidate_source"].str.contains("current_squad").all(),
          f"GW{gw}: retained squad rows tagged current_squad")

    for position in FPL_POSITIONS:
        pos_pool = pool[pool["position"] == position]
        squad_count = len(set(squad) & set(df[df["position"] == position]["player_id"]))
        limit = DEFAULT_PREDICTION_TOP_K[position] + DEFAULT_FORM_TOP_K[position] + squad_count
        check(len(pos_pool) <= limit, f"GW{gw} {position}: pool size ({len(pos_pool)}) within limit ({limit})")


def test_double_gameweek() -> None:
    df = build_canonical_predictions(gameweek=DGW_GAMEWEEK)
    dgw_players = df[df.duplicated("player_id", keep=False)]["player_id"].unique()
    check(len(dgw_players) > 0, f"GW{DGW_GAMEWEEK}: has real double-gameweek players ({len(dgw_players)})")

    sample_id = dgw_players[0]
    raw_rows = df[df["player_id"] == sample_id]
    expected_sum = raw_rows["predicted_points"].sum()

    pool = build_candidate_pool(df, target_gameweek=DGW_GAMEWEEK, current_squad_ids=[int(sample_id)])
    collapsed_row = pool[pool["player_id"] == sample_id]
    check(len(collapsed_row) == 1, f"DGW player {sample_id} appears once in the pool, not twice")
    check(abs(collapsed_row["predicted_points"].iloc[0] - expected_sum) < 1e-6,
          f"DGW player {sample_id}: predicted_points is the sum of both fixtures")
    check(collapsed_row["fixture_count"].iloc[0] == 2, f"DGW player {sample_id}: fixture_count is 2")

    non_dgw = pool[~pool["player_id"].isin(dgw_players)]
    if len(non_dgw):
        check((non_dgw["fixture_count"] == 1).all(), "non-DGW players keep fixture_count 1")


def test_form_no_future_leakage(gw: int, df: pd.DataFrame) -> None:
    raw = pd.read_csv(DATA_RAW_DIR / "player_match_stats.csv", engine="python")
    raw = raw[raw["season"].astype(str) == "2025-26"].copy()
    raw["total_points"] = pd.to_numeric(raw["total_points"], errors="coerce").fillna(0)
    raw = raw.sort_values(["player_id", "gameweek", "fixture_id"])

    sample = df.dropna(subset=["form_avg3"]).sample(n=min(5, len(df)), random_state=gw)
    mismatches = 0
    for _, row in sample.iterrows():
        history = raw[(raw["player_id"] == row["player_id"]) & (raw["gameweek"] < gw)]
        if history.empty:
            continue
        expected = history.tail(3)["total_points"].mean()
        if abs(expected - row["form_avg3"]) > 1e-6:
            mismatches += 1
    check(mismatches == 0, f"GW{gw}: form_avg3 matches independent recomputation from GW<{gw} only")


def test_pipeline_driver(gw: int) -> None:
    pool = get_candidates(target_gameweek=gw, current_squad_ids=[])
    check(len(pool) > 0, f"GW{gw}: get_candidates() driver returns a non-empty pool")
    check("candidate_source" in pool.columns, f"GW{gw}: driver output has candidate_source")


def main() -> None:
    for gw in TEST_GAMEWEEKS:
        df = test_schema_and_ranking(gw)
        test_candidate_pool(gw, df)
        test_form_no_future_leakage(gw, df)
        test_pipeline_driver(gw)

    test_double_gameweek()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

"""Tests for the historical squad/environment layer.

Run with:
    python rl_decision_layer/tests/test_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from rl_decision_layer.environment.historical_env import HistoricalEnv
from rl_decision_layer.environment.squad import (
    MAX_PER_CLUB,
    POSITION_COUNTS,
    SQUAD_SIZE,
    build_starting_squad,
    validate_squad,
)
from rl_decision_layer.predictions.interface import DATA_RAW_DIR, build_canonical_predictions

START_GAMEWEEKS = [10, 25]

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_starting_squad(gw: int) -> list[int]:
    df = build_canonical_predictions(gameweek=gw)
    squad = build_starting_squad(df)

    check(len(squad) == SQUAD_SIZE, f"GW{gw}: starting squad has {SQUAD_SIZE} players ({len(squad)})")
    check(len(set(squad)) == len(squad), f"GW{gw}: no duplicate players in starting squad")

    errors = validate_squad(df, squad)
    check(errors == [], f"GW{gw}: starting squad is legal ({errors})")

    rows = df[df["player_id"].isin(squad)]
    club_counts = rows["team_id"].value_counts()
    check((club_counts <= MAX_PER_CLUB).all(), f"GW{gw}: no club has more than {MAX_PER_CLUB} players")

    for position, n in POSITION_COUNTS.items():
        actual = (rows["position"] == position).sum()
        check(actual == n, f"GW{gw}: {position} count is {actual}, expected {n}")

    return squad


def test_reset_and_decision_state(gw: int, squad: list[int]) -> None:
    env = HistoricalEnv()
    state = env.reset(start_gameweek=gw, squad_ids=squad, bank=0.0, free_transfers=1)

    check(state.gameweek == gw, f"GW{gw}: decision state has the right gameweek")
    check(set(state.squad_ids) == set(squad), f"GW{gw}: decision state carries the full squad")
    check(len(state.candidates) > 0, f"GW{gw}: candidates generated through the env ({len(state.candidates)} rows)")
    check("candidate_source" in state.candidates.columns, f"GW{gw}: candidates have candidate_source")

    no_actuals = "actual_points" not in state.candidates.columns and "total_points" not in state.candidates.columns
    check(no_actuals, f"GW{gw}: decision-time candidates contain no actual-points column")
    check(not hasattr(state, "actual_points"), f"GW{gw}: DecisionState itself has no actual_points field")


def test_step_and_outcome(gw: int, squad: list[int]) -> None:
    env = HistoricalEnv()
    env.reset(start_gameweek=gw, squad_ids=squad)
    outcome, next_state = env.step()

    check(outcome.gameweek == gw, f"GW{gw}: outcome corresponds to the gameweek that was actually played")
    check(set(outcome.actual_points.keys()) == set(squad), f"GW{gw}: outcome has points for every squad player")

    raw = pd.read_csv(DATA_RAW_DIR / "player_match_stats.csv", engine="python")
    raw = raw[raw["season"].astype(str) == "2025-26"].copy()
    raw["total_points"] = pd.to_numeric(raw["total_points"], errors="coerce").fillna(0)
    sample_id = squad[0]
    expected = raw[(raw["player_id"] == sample_id) & (raw["gameweek"] == gw)]["total_points"].sum()
    check(abs(outcome.actual_points[sample_id] - expected) < 1e-6,
          f"GW{gw}: player {sample_id}'s outcome points match raw match stats independently")

    check(next_state.gameweek == gw + 1, f"GW{gw}: env advances to GW{gw + 1} after step()")
    check(env.gameweek == gw + 1, f"GW{gw}: env's internal gameweek counter also advanced")

    # nothing about gw+1 should have influenced the outcome we just calculated for gw
    check(outcome.gameweek != next_state.gameweek, f"GW{gw}: outcome gameweek and next decision gameweek are distinct")


def main() -> None:
    for gw in START_GAMEWEEKS:
        squad = test_starting_squad(gw)
        test_reset_and_decision_state(gw, squad)
        test_step_and_outcome(gw, squad)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

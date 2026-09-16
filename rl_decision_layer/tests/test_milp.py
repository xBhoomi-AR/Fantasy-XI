"""Tests for the squad-selection MILP.

Run with:
    python rl_decision_layer/tests/test_milp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from rl_decision_layer.candidates.pipeline import get_candidates
from rl_decision_layer.environment.squad import BUDGET, MAX_PER_CLUB, POSITION_COUNTS, SQUAD_SIZE
from rl_decision_layer.optimization.squad_milp import select_squad

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def make_pool(counts: dict[str, int], num_teams: int = 6) -> pd.DataFrame:
    """Small synthetic candidate pool, spread evenly across num_teams clubs
    so the 3-per-club constraint doesn't accidentally make it infeasible."""
    rows = []
    pid = 1
    for position, n in counts.items():
        for i in range(n):
            rows.append({
                "player_id": pid,
                "position": position,
                "team_id": (pid % num_teams) + 1,
                "price": 40 + (pid % 5) * 3,
                "predicted_points": 2.0 + (pid % 7) * 0.5,
            })
            pid += 1
    return pd.DataFrame(rows)


def test_feasible_pool() -> None:
    pool = make_pool({"GK": 4, "DEF": 10, "MID": 10, "FWD": 6})
    result = select_squad(pool)

    check(result.status == "Optimal", "synthetic pool: solver finds a solution")
    check(len(result.selected_ids) == SQUAD_SIZE, f"synthetic pool: exactly {SQUAD_SIZE} players selected")

    selected_rows = pool[pool["player_id"].isin(result.selected_ids)]
    for position, expected in POSITION_COUNTS.items():
        actual = (selected_rows["position"] == position).sum()
        check(actual == expected, f"synthetic pool: {position} count is {actual}, expected {expected}")

    club_counts = selected_rows["team_id"].value_counts()
    check((club_counts <= MAX_PER_CLUB).all(), "synthetic pool: no club has more than 3 selected players")

    check(result.total_cost <= BUDGET, f"synthetic pool: total cost {result.total_cost} within budget {BUDGET}")

    check(set(result.selected_ids) <= set(pool["player_id"]), "synthetic pool: selected players all came from the pool")

    expected_objective = selected_rows.drop_duplicates("player_id")["predicted_points"].sum()
    check(abs(result.objective_value - expected_objective) < 1e-6,
          "synthetic pool: objective value equals the sum of selected players' predicted_points")


def test_deterministic() -> None:
    pool = make_pool({"GK": 4, "DEF": 10, "MID": 10, "FWD": 6})
    r1 = select_squad(pool)
    r2 = select_squad(pool)
    check(sorted(r1.selected_ids) == sorted(r2.selected_ids), "same input produces the same squad on repeated runs")


def test_infeasible_not_enough_goalkeepers() -> None:
    pool = make_pool({"GK": 1, "DEF": 10, "MID": 10, "FWD": 6})
    result = select_squad(pool)
    check(result.status != "Optimal", "pool with only 1 GK is reported infeasible, not a broken squad")
    check(result.selected_ids == [], "infeasible result has no selected players")


def test_infeasible_budget_too_low() -> None:
    pool = make_pool({"GK": 4, "DEF": 10, "MID": 10, "FWD": 6})
    result = select_squad(pool, budget=50)  # far below what 15 players cost at price >= 40 each
    check(result.status != "Optimal", "pool with an unaffordable budget is reported infeasible")


def test_already_owned_is_tracked() -> None:
    pool = make_pool({"GK": 4, "DEF": 10, "MID": 10, "FWD": 6})
    result = select_squad(pool, current_squad_ids=[1, 2, 3])
    check(set(result.already_owned) <= set(result.selected_ids), "already_owned is a subset of the selected squad")


def test_real_gameweek(gw: int) -> None:
    pool = get_candidates(gw, current_squad_ids=[])
    result = select_squad(pool)

    print(f"\nGW{gw}: {len(pool)} candidates -> status={result.status}, "
          f"cost={result.total_cost:.1f}, objective={result.objective_value:.2f}, "
          f"remaining_budget={result.remaining_budget:.1f}")
    print(f"  by position: {({k: len(v) for k, v in result.by_position.items()})}")

    check(result.status == "Optimal", f"GW{gw}: real candidate pool is solvable")
    check(len(result.selected_ids) == SQUAD_SIZE, f"GW{gw}: exactly {SQUAD_SIZE} players selected")

    selected_rows = pool[pool["player_id"].isin(result.selected_ids)].drop_duplicates("player_id")
    for position, expected in POSITION_COUNTS.items():
        actual = (selected_rows["position"] == position).sum()
        check(actual == expected, f"GW{gw}: {position} count is {actual}, expected {expected}")

    club_counts = selected_rows["team_id"].value_counts()
    check((club_counts <= MAX_PER_CLUB).all(), f"GW{gw}: no club exceeds {MAX_PER_CLUB} players")
    check(result.total_cost <= BUDGET, f"GW{gw}: total cost within budget")
    check(set(result.selected_ids) <= set(pool["player_id"]), f"GW{gw}: selected players all came from the candidate pool")


def main() -> None:
    test_feasible_pool()
    test_deterministic()
    test_infeasible_not_enough_goalkeepers()
    test_infeasible_budget_too_low()
    test_already_owned_is_tracked()

    for gw in [5, 15, 26, 35]:  # spread across the season, 26 is a real double gameweek
        test_real_gameweek(gw)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

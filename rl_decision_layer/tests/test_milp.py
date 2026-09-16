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


def test_transfer_aware_no_slack_keeps_squad() -> None:
    pool = make_pool({"GK": 4, "DEF": 10, "MID": 10, "FWD": 6})
    fresh = select_squad(pool)
    # budget = exactly what the current squad is worth, no bank - nothing to upgrade with
    result = select_squad(pool, current_squad_ids=fresh.selected_ids, bank=0.0, free_transfers=1)
    check(result.status == "Optimal", "transfer-aware: solvable with no slack")
    check(result.transfers_made == 0, "transfer-aware: no transfers made when there's no budget slack")
    check(result.hits == 0, "transfer-aware: no hits when no transfers are made")


def test_transfer_aware_more_free_transfers_never_hurts() -> None:
    pool = make_pool({"GK": 4, "DEF": 10, "MID": 10, "FWD": 6})
    fresh = select_squad(pool)
    low_ft = select_squad(pool, current_squad_ids=fresh.selected_ids, bank=100.0, free_transfers=1)
    high_ft = select_squad(pool, current_squad_ids=fresh.selected_ids, bank=100.0, free_transfers=5)
    check(low_ft.status == "Optimal" and high_ft.status == "Optimal", "transfer-aware: both solvable")
    check(high_ft.objective_value >= low_ft.objective_value - 1e-6,
          "transfer-aware: more free transfers never makes the objective worse")


def test_transfer_aware_real_gameweek(start_gw: int, target_gw: int) -> None:
    from rl_decision_layer.environment.squad import build_starting_squad, squad_value
    from rl_decision_layer.predictions.interface import build_canonical_predictions

    squad = build_starting_squad(build_canonical_predictions(gameweek=start_gw))
    pool = get_candidates(target_gw, current_squad_ids=squad)
    # build_starting_squad is deliberately the cheapest legal squad (see
    # environment/squad.py), so it has very little value to sell - needs a
    # real bank cushion here or this can genuinely go infeasible
    bank = 200.0
    result = select_squad(pool, current_squad_ids=squad, bank=bank, free_transfers=2)

    print(f"\nGW{target_gw} transfer-aware (from GW{start_gw} squad): transfers={result.transfers_made} "
          f"hits={result.hits} cost={result.total_cost:.1f} objective={result.objective_value:.2f} status={result.status}")

    check(result.status == "Optimal", f"GW{target_gw} transfer-aware: solvable")
    selected_rows = pool[pool["player_id"].isin(result.selected_ids)].drop_duplicates("player_id")
    for position, expected in POSITION_COUNTS.items():
        actual = (selected_rows["position"] == position).sum()
        check(actual == expected, f"GW{target_gw} transfer-aware: {position} count correct")
    club_counts = selected_rows["team_id"].value_counts()
    check((club_counts <= MAX_PER_CLUB).all(), f"GW{target_gw} transfer-aware: club limit respected")

    available = bank + squad_value(pool, squad)
    check(result.total_cost <= available + 1e-6, f"GW{target_gw} transfer-aware: cost within bank+squad-value budget")


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
    test_transfer_aware_no_slack_keeps_squad()
    test_transfer_aware_more_free_transfers_never_hurts()

    for gw in [5, 15, 26, 35]:  # spread across the season, 26 is a real double gameweek
        test_real_gameweek(gw)

    for start_gw, target_gw in [(19, 20), (25, 26)]:  # 26 is a real double gameweek
        test_transfer_aware_real_gameweek(start_gw, target_gw)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

"""Tests for starting XI / captain selection.

Run with:
    python rl_decision_layer/tests/test_starting_xi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.candidates.pipeline import get_candidates
from rl_decision_layer.optimization.squad_milp import select_squad
from rl_decision_layer.optimization.starting_xi import (
    MAX_DEF, MAX_FWD, MAX_MID, MIN_DEF, MIN_FWD, MIN_MID, XI_SIZE, pick_starting_xi,
)

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_real_gameweek(gw: int) -> None:
    pool = get_candidates(gw, current_squad_ids=[])
    squad = select_squad(pool)
    squad_rows = pool[pool["player_id"].isin(squad.selected_ids)]

    xi = pick_starting_xi(squad_rows)
    print(f"\nGW{gw}: captain={xi.captain_id} vice={xi.vice_captain_id} predicted_points={xi.predicted_points:.2f}")

    check(xi.status == "Optimal", f"GW{gw}: starting XI is solvable from a legal 15-man squad")
    check(len(xi.starting_ids) == XI_SIZE, f"GW{gw}: exactly {XI_SIZE} starters")
    check(len(xi.bench_ids) == 4, f"GW{gw}: 4 players benched")
    check(set(xi.starting_ids).isdisjoint(xi.bench_ids), f"GW{gw}: no player is both starting and benched")
    check(set(xi.starting_ids) <= set(squad.selected_ids), f"GW{gw}: starters all came from the selected squad")

    counts = squad_rows[squad_rows["player_id"].isin(xi.starting_ids)]["position"].value_counts()
    check(counts.get("GK", 0) == 1, f"GW{gw}: exactly 1 starting GK")
    check(MIN_DEF <= counts.get("DEF", 0) <= MAX_DEF, f"GW{gw}: DEF count {counts.get('DEF', 0)} within {MIN_DEF}-{MAX_DEF}")
    check(MIN_MID <= counts.get("MID", 0) <= MAX_MID, f"GW{gw}: MID count {counts.get('MID', 0)} within {MIN_MID}-{MAX_MID}")
    check(MIN_FWD <= counts.get("FWD", 0) <= MAX_FWD, f"GW{gw}: FWD count {counts.get('FWD', 0)} within {MIN_FWD}-{MAX_FWD}")

    check(xi.captain_id in xi.starting_ids, f"GW{gw}: captain is a starter")
    check(xi.vice_captain_id in xi.starting_ids, f"GW{gw}: vice-captain is a starter")
    check(xi.captain_id != xi.vice_captain_id, f"GW{gw}: captain and vice-captain are different players")

    captain_points = squad_rows.loc[squad_rows["player_id"] == xi.captain_id, "predicted_points"].iloc[0]
    starter_points = squad_rows[squad_rows["player_id"].isin(xi.starting_ids)]["predicted_points"]
    check(captain_points == starter_points.max(), f"GW{gw}: captain has the highest predicted_points among starters")


def test_deterministic() -> None:
    pool = get_candidates(20, current_squad_ids=[])
    squad = select_squad(pool)
    squad_rows = pool[pool["player_id"].isin(squad.selected_ids)]
    r1 = pick_starting_xi(squad_rows)
    r2 = pick_starting_xi(squad_rows)
    check(sorted(r1.starting_ids) == sorted(r2.starting_ids), "same squad produces the same starting XI on repeated runs")
    check(r1.captain_id == r2.captain_id, "same squad produces the same captain on repeated runs")


def main() -> None:
    for gw in [5, 20, 35]:
        test_real_gameweek(gw)
    test_deterministic()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

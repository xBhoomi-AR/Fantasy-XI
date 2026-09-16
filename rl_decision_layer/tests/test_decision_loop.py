"""Integration test: HistoricalEnv + the MILP decide() bridge, stepping
through real consecutive historical gameweeks end to end.

Run with:
    python rl_decision_layer/tests/test_decision_loop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.environment.historical_env import HistoricalEnv
from rl_decision_layer.environment.squad import MAX_PER_CLUB, SQUAD_SIZE, build_starting_squad, validate_squad
from rl_decision_layer.optimization.squad_milp import decide
from rl_decision_layer.predictions.interface import build_canonical_predictions

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def run_loop(start_gw: int, num_steps: int):
    squad = build_starting_squad(build_canonical_predictions(gameweek=start_gw))
    env = HistoricalEnv()
    state = env.reset(start_gameweek=start_gw, squad_ids=squad, bank=20.0, free_transfers=1)

    for _ in range(num_steps):
        gw = state.gameweek

        # leakage: decision state must not carry this gameweek's actual result
        check(not hasattr(state, "actual_points"), f"GW{gw}: decision state has no actual_points field")
        check(not any("actual" in c.lower() for c in state.candidates.columns),
              f"GW{gw}: candidates carry predictions only, no actual-result column")

        decision = decide(state)
        check(decision.status == "Optimal", f"GW{gw}: MILP found a legal decision")

        errors = validate_squad(state.candidates, decision.selected_ids)
        check(errors == [], f"GW{gw}: resulting squad is structurally legal ({errors})")

        club_rows = state.candidates[state.candidates["player_id"].isin(decision.selected_ids)].drop_duplicates("player_id")
        check((club_rows["team_id"].value_counts() <= MAX_PER_CLUB).all(), f"GW{gw}: club limit respected")

        outcome, next_state = env.step(decision.selected_ids)

        check(outcome.gameweek == gw, f"GW{gw}: outcome corresponds to the gameweek just decided")
        check(set(outcome.actual_points.keys()) == set(decision.selected_ids), f"GW{gw}: outcome covers exactly the selected squad")
        check(next_state.gameweek == gw + 1, f"GW{gw}: environment advanced to GW{gw + 1}")

        state = next_state

    return state


def test_two_gameweek_loop() -> None:
    final_state = run_loop(start_gw=19, num_steps=2)
    check(final_state.gameweek == 21, "loop ends at the expected gameweek")


def test_double_gameweek_outcome() -> None:
    # step from GW25 into GW26, a real double gameweek, and confirm the
    # environment handles it correctly through a real decision
    squad = build_starting_squad(build_canonical_predictions(gameweek=25))
    env = HistoricalEnv()
    state = env.reset(start_gameweek=25, squad_ids=squad, bank=50.0, free_transfers=2)

    decision = decide(state)
    check(decision.status == "Optimal", "GW25: MILP decision found")
    outcome, next_state = env.step(decision.selected_ids)
    check(outcome.gameweek == 25, "GW25: outcome is for GW25, not GW26")
    check(next_state.gameweek == 26, "GW25: env moved into GW26, the double gameweek")

    dgw_decision = decide(next_state)
    check(dgw_decision.status == "Optimal", "GW26 (DGW): MILP decision found")
    dgw_outcome, _ = env.step(dgw_decision.selected_ids)
    check(dgw_outcome.gameweek == 26, "GW26 (DGW): outcome gameweek correct")
    check(len(dgw_outcome.actual_points) == SQUAD_SIZE, "GW26 (DGW): outcome has points for all 15 squad players")


def test_deterministic() -> None:
    squad = build_starting_squad(build_canonical_predictions(gameweek=19))

    env1 = HistoricalEnv()
    d1 = decide(env1.reset(20, squad, bank=20.0, free_transfers=1))

    env2 = HistoricalEnv()
    d2 = decide(env2.reset(20, squad, bank=20.0, free_transfers=1))

    check(sorted(d1.selected_ids) == sorted(d2.selected_ids), "identical state produces an identical MILP decision")


def main() -> None:
    test_two_gameweek_loop()
    test_double_gameweek_outcome()
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

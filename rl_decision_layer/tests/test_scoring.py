"""Tests for actual-points scoring and the reward.

Run with:
    python rl_decision_layer/tests/test_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.environment.historical_env import Outcome
from rl_decision_layer.optimization.scoring import calculate_reward, score_outcome
from rl_decision_layer.optimization.starting_xi import StartingXIResult

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_captain_doubles() -> None:
    outcome = Outcome(gameweek=10, squad_ids=[1, 2, 3], actual_points={1: 10.0, 2: 5.0, 3: 0.0})
    xi = StartingXIResult(status="Optimal", starting_ids=[1, 2], bench_ids=[3],
                           captain_id=1, vice_captain_id=2, predicted_points=15.0)

    scored = score_outcome(outcome, xi)
    check(scored.starting_points == 15.0, "starting_points is the sum of starters only, bench excluded")
    check(scored.captain_points == 10.0, "captain_points is the captain's own score")
    check(scored.total_points == 25.0, "total_points counts the captain's points twice (10+5+10)")


def test_bench_player_not_counted() -> None:
    outcome = Outcome(gameweek=10, squad_ids=[1, 2, 3], actual_points={1: 10.0, 2: 5.0, 3: 20.0})
    xi = StartingXIResult(status="Optimal", starting_ids=[1, 2], bench_ids=[3],
                           captain_id=1, vice_captain_id=2, predicted_points=15.0)
    scored = score_outcome(outcome, xi)
    check(scored.total_points == 25.0, "a benched player's huge score (20) is not counted")


def test_reward_subtracts_hits() -> None:
    outcome = Outcome(gameweek=10, squad_ids=[1, 2], actual_points={1: 10.0, 2: 5.0})
    xi = StartingXIResult(status="Optimal", starting_ids=[1, 2], bench_ids=[],
                           captain_id=1, vice_captain_id=2, predicted_points=15.0)
    scored = score_outcome(outcome, xi)  # total = 10+5+10 = 25

    check(calculate_reward(scored, hits=0) == 25.0, "no hits: reward equals total_points")
    check(calculate_reward(scored, hits=2) == 17.0, "2 hits: reward is total_points minus 2*4")


def main() -> None:
    test_captain_doubles()
    test_bench_player_not_counted()
    test_reward_subtracts_hits()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

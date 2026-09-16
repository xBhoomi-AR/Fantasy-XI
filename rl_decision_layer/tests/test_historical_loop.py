"""Smoke test for the full chronological MILP-only backtest loop.

Run with:
    python rl_decision_layer/tests/test_historical_loop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.environment.squad import MAX_PER_CLUB, POSITION_COUNTS, SQUAD_SIZE
from rl_decision_layer.historical_loop import run_backtest
from rl_decision_layer.predictions.interface import build_canonical_predictions

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_short_run() -> None:
    """A handful of gameweeks, checked in detail."""
    logs = run_backtest(start_gameweek=10, num_gameweeks=5, initial_bank=0.0, free_transfers=1)
    check(len(logs) == 5, f"5-gameweek run completes all 5 gameweeks ({len(logs)} logged)")

    for log in logs:
        check(len(log.squad_ids) == SQUAD_SIZE, f"GW{log.gameweek}: squad has {SQUAD_SIZE} players")
        check(len(log.starting_ids) == 11, f"GW{log.gameweek}: 11 starters")
        check(log.captain_id in log.starting_ids, f"GW{log.gameweek}: captain is a starter")
        check(log.vice_captain_id in log.starting_ids, f"GW{log.gameweek}: vice-captain is a starter")
        check(log.captain_id != log.vice_captain_id, f"GW{log.gameweek}: captain != vice-captain")

        df = build_canonical_predictions(gameweek=log.gameweek)
        squad_rows = df[df["player_id"].isin(log.squad_ids)].drop_duplicates("player_id")
        for position, expected in POSITION_COUNTS.items():
            actual = (squad_rows["position"] == position).sum()
            check(actual == expected, f"GW{log.gameweek}: {position} count correct")
        check((squad_rows["team_id"].value_counts() <= MAX_PER_CLUB).all(), f"GW{log.gameweek}: club limit respected")

    # gameweeks are consecutive, nothing skipped
    gws = [log.gameweek for log in logs]
    check(gws == list(range(10, 15)), f"gameweeks are consecutive: {gws}")


def test_double_gameweek_included() -> None:
    """Run across GW24-28, which includes the real GW26 double gameweek."""
    logs = run_backtest(start_gameweek=24, num_gameweeks=5, initial_bank=20.0, free_transfers=1)
    check(len(logs) == 5, f"run spanning the DGW completes ({len(logs)} of 5 gameweeks)")
    dgw_log = next((log for log in logs if log.gameweek == 26), None)
    check(dgw_log is not None, "GW26 (the double gameweek) is present in the run")
    if dgw_log:
        check(dgw_log.actual_points >= 0, "GW26: actual points calculated without error")


def test_reward_matches_manual_calculation() -> None:
    from rl_decision_layer.environment.historical_env import HistoricalEnv
    from rl_decision_layer.environment.squad import build_starting_squad
    from rl_decision_layer.optimization.scoring import calculate_reward, score_outcome
    from rl_decision_layer.optimization.squad_milp import decide
    from rl_decision_layer.optimization.starting_xi import pick_starting_xi

    squad = build_starting_squad(build_canonical_predictions(gameweek=15))
    env = HistoricalEnv()
    state = env.reset(15, squad, bank=10.0, free_transfers=1)
    decision = decide(state)
    squad_rows = state.candidates[state.candidates["player_id"].isin(decision.selected_ids)]
    xi = pick_starting_xi(squad_rows)
    outcome, _ = env.step(decision.selected_ids, new_bank=decision.remaining_budget)
    scored = score_outcome(outcome, xi)
    reward = calculate_reward(scored, decision.hits)

    logs = run_backtest(start_gameweek=15, num_gameweeks=1, initial_squad=squad, initial_bank=10.0, free_transfers=1)
    check(abs(logs[0].reward - reward) < 1e-6, "run_backtest's reward matches a manually assembled decide/step/score pipeline")


def main() -> None:
    test_short_run()
    test_double_gameweek_included()
    test_reward_matches_manual_calculation()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

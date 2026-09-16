"""Tests for free-transfer rollover in HistoricalEnv.step().

Run with:
    python rl_decision_layer/tests/test_free_transfers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.environment.historical_env import HistoricalEnv
from rl_decision_layer.environment.squad import FREE_TRANSFER_CAP, build_starting_squad
from rl_decision_layer.optimization.squad_milp import decide
from rl_decision_layer.predictions.interface import build_canonical_predictions

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def make_env(gw: int, free_transfers: int = 1):
    squad = build_starting_squad(build_canonical_predictions(gameweek=gw))
    env = HistoricalEnv()
    state = env.reset(gw, squad, bank=0.0, free_transfers=free_transfers)
    return env, state, squad


def test_unused_transfer_rolls_forward() -> None:
    env, state, squad = make_env(10, free_transfers=1)
    _, next_state = env.step(new_squad_ids=squad)  # no change = 0 transfers used
    check(next_state.free_transfers == 2, f"1 unused transfer rolls to 2 (got {next_state.free_transfers})")


def test_accumulation_over_several_weeks() -> None:
    env, state, squad = make_env(10, free_transfers=1)
    for expected in [2, 3, 4]:
        _, state = env.step(new_squad_ids=squad)
        check(state.free_transfers == expected, f"accumulates to {expected} (got {state.free_transfers})")


def test_capped_at_five() -> None:
    env, state, squad = make_env(10, free_transfers=5)
    for _ in range(3):
        _, state = env.step(new_squad_ids=squad)
    check(state.free_transfers == FREE_TRANSFER_CAP, f"never exceeds the cap of {FREE_TRANSFER_CAP} (got {state.free_transfers})")


def test_transfers_consume_free_transfers() -> None:
    env, state, squad = make_env(10, free_transfers=3)
    new_squad = squad[2:] + [1, 2]  # 2 players swapped = 2 transfers, within the 3 free
    _, next_state = env.step(new_squad_ids=new_squad)
    check(next_state.free_transfers == 2, f"using 2 of 3 free transfers leaves 1, rolls to 2 (got {next_state.free_transfers})")


def test_transfers_beyond_free_become_hits() -> None:
    # this is squad_milp's own responsibility, not the environment's - confirm
    # the two stay consistent for the same transfer count
    squad = build_starting_squad(build_canonical_predictions(gameweek=20))
    env = HistoricalEnv()
    state = env.reset(20, squad, bank=200.0, free_transfers=1)
    decision = decide(state)
    check(decision.status == "Optimal", "transfer-aware decision solvable")

    _, next_state = env.step(decision.selected_ids, new_bank=decision.remaining_budget)
    expected_remaining_free = max(0, 1 - min(decision.transfers_made, 1)) + 1
    expected_remaining_free = min(FREE_TRANSFER_CAP, expected_remaining_free)
    check(next_state.free_transfers == expected_remaining_free,
          f"free_transfers after a real MILP decision matches the rollover formula "
          f"(transfers_made={decision.transfers_made}, hits={decision.hits}, "
          f"expected={expected_remaining_free}, got={next_state.free_transfers})")


def test_using_all_transfers_resets_to_one() -> None:
    env, state, squad = make_env(10, free_transfers=3)
    new_squad = squad[3:] + [1, 2, 3]  # exactly 3 transfers, uses all 3 free
    _, next_state = env.step(new_squad_ids=new_squad)
    check(next_state.free_transfers == 1, f"using exactly all free transfers resets to 1 next week (got {next_state.free_transfers})")


def test_bank_and_free_transfers_propagate_together() -> None:
    env, state, squad = make_env(10, free_transfers=2)
    _, next_state = env.step(new_squad_ids=squad, new_bank=17.5)
    check(next_state.bank == 17.5, "bank propagates alongside free_transfers")
    check(next_state.free_transfers == 3, "free_transfers still accrues correctly at the same time")


def test_three_consecutive_gameweeks() -> None:
    # use clearly-fake negative IDs for "new" players so the transfer count
    # is unambiguous regardless of what's already in the real squad
    env, state, squad = make_env(10, free_transfers=1)
    check(state.free_transfers == 1, "GW10 starts at 1")

    _, state = env.step(new_squad_ids=squad, new_bank=5.0)  # 0 transfers
    check(state.gameweek == 11, "advanced to GW11")
    check(state.free_transfers == 2 and state.bank == 5.0, f"GW11: free_transfers=2, bank=5.0 (got {state.free_transfers}, {state.bank})")

    squad_v2 = squad[2:] + [-1, -2]  # 2 transfers, uses both free ones
    _, state = env.step(new_squad_ids=squad_v2, new_bank=2.0)
    check(state.gameweek == 12, "advanced to GW12")
    check(state.free_transfers == 1 and state.bank == 2.0, f"GW12: free_transfers=1, bank=2.0 (got {state.free_transfers}, {state.bank})")

    squad_v3 = squad_v2[1:] + [-3]  # 1 transfer, uses the 1 free transfer available
    _, state = env.step(new_squad_ids=squad_v3, new_bank=8.0)
    check(state.gameweek == 13, "advanced to GW13")
    check(state.free_transfers == 1 and state.bank == 8.0, f"GW13: free_transfers=1, bank=8.0 (got {state.free_transfers}, {state.bank})")


def main() -> None:
    test_unused_transfer_rolls_forward()
    test_accumulation_over_several_weeks()
    test_capped_at_five()
    test_transfers_consume_free_transfers()
    test_transfers_beyond_free_become_hits()
    test_using_all_transfers_resets_to_one()
    test_bank_and_free_transfers_propagate_together()
    test_three_consecutive_gameweeks()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

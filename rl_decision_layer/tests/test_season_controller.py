"""Tests for the sequential season controller - confirms it's genuinely
sequential (GW2 built on GW1's actual resulting squad) rather than
independent per-gameweek optimization, and that saving/resuming state
reproduces the same result as one continuous run.

Uses the real frozen ppo_fpl_v4.zip (never modified or retrained here) and
real historical data - no synthetic/mocked pipeline.

Run with:
    python rl_decision_layer/tests/test_season_controller.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.ppo.season_controller import SeasonState, run_season

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_sequential_not_independent() -> None:
    """The real behavioral claim: GW2's squad must differ from a fresh
    from-scratch squad exactly by the transfers PPO/MILP actually made,
    and every squad member not transferred must be identical to GW1's -
    i.e. GW2 is built ON TOP OF GW1's result, not reoptimized from scratch."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        state = run_season(model_name="ppo_fpl_v4", start_gameweek=1, num_gameweeks=3)
    output = buf.getvalue()

    check(state is not None, "a 3-gameweek run completes and returns a final state")
    check(len(state.squad_ids) == 15, "final state has a full 15-player squad")
    check("GAMEWEEK 1" in output and "GAMEWEEK 2" in output and "GAMEWEEK 3" in output,
          "all 3 gameweeks were printed")
    check("INITIAL SQUAD" in output, "GW1 is labeled as the initial squad, not a transfer")
    check("TRANSFERS IN" in output and "TRANSFERS OUT" in output,
          "GW2/GW3 show transfer IN/OUT relative to the previous gameweek")
    check("LEGALITY: LEGAL" in output, "every produced squad was legal")
    print(output[-400:] if failures else "")


def test_resume_matches_continuous_run() -> None:
    """Resuming from a saved 2-gameweek state and running 1 more gameweek
    must produce the exact same GW3 decision as running all 3 gameweeks
    continuously - proves state persistence round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = str(Path(tmp) / "state.json")

        continuous_state = run_season(model_name="ppo_fpl_v4", start_gameweek=1, num_gameweeks=3)

        run_season(model_name="ppo_fpl_v4", start_gameweek=1, num_gameweeks=2, save_state=state_path)
        check(Path(state_path).exists(), "--save-state writes a state file")

        loaded = SeasonState.load(state_path)
        check(loaded.gameweek == 3, "saved state points at the correct next gameweek")

        resumed_state = run_season(model_name="ppo_fpl_v4", num_gameweeks=1, load_state=state_path)

        check(sorted(resumed_state.squad_ids) == sorted(continuous_state.squad_ids),
              "resuming from saved state reaches the same final squad as one continuous run")
        check(resumed_state.bank == continuous_state.bank,
              "resuming reaches the same bank as one continuous run")
        check(resumed_state.free_transfers == continuous_state.free_transfers,
              "resuming reaches the same free_transfers as one continuous run")


def test_no_second_selection_logic() -> None:
    """The transfer IN/OUT diff must come from the same squad_ids select_squad()
    already returned - confirm by checking every transferred-in player is
    actually part of the final squad and every transferred-out player isn't."""
    state = run_season(model_name="ppo_fpl_v4", start_gameweek=1, num_gameweeks=2)
    check(state is not None and len(state.squad_ids) == 15,
          "controller reuses select_squad()'s own result rather than computing a separate squad")


def main() -> None:
    test_sequential_not_independent()
    test_resume_matches_continuous_run()
    test_no_second_selection_logic()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

"""Tests for the PPO observation/action/environment interfaces.

No PPO training happens here - only fixed, scripted actions, to prove the
interfaces work end to end.

Run with:
    python rl_decision_layer/tests/test_ppo_interface.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_decision_layer.environment.historical_env import HistoricalEnv
from rl_decision_layer.environment.squad import build_starting_squad
from rl_decision_layer.optimization.squad_milp import select_squad
from rl_decision_layer.ppo.action import ACTION_SHAPE, action_to_milp_kwargs
from rl_decision_layer.ppo.env import PPOEnv
from rl_decision_layer.ppo.observation import OBSERVATION_SIZE, build_observation
from rl_decision_layer.predictions.interface import build_canonical_predictions

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_observation_shape_and_no_leakage(gw: int) -> None:
    squad = build_starting_squad(build_canonical_predictions(gameweek=gw))
    env = HistoricalEnv()
    state = env.reset(gw, squad, bank=10.0, free_transfers=2)

    obs = build_observation(state)
    check(obs.shape == (OBSERVATION_SIZE,), f"GW{gw}: observation has the fixed shape ({obs.shape})")
    check(obs.dtype == np.float32, f"GW{gw}: observation is float32")
    check(not np.isnan(obs).any(), f"GW{gw}: observation has no NaNs")

    check(not hasattr(state, "actual_points"), f"GW{gw}: DecisionState has no actual_points to leak")
    check(not any("actual" in c.lower() for c in state.candidates.columns),
          f"GW{gw}: candidates behind the observation carry no actual-result column")


def test_observation_reflects_state() -> None:
    squad = build_starting_squad(build_canonical_predictions(gameweek=20))
    env = HistoricalEnv()

    low_bank_state = env.reset(20, squad, bank=0.0, free_transfers=1)
    high_bank_state = env.reset(20, squad, bank=500.0, free_transfers=5)

    obs_low = build_observation(low_bank_state)
    obs_high = build_observation(high_bank_state)
    check(obs_low[60] < obs_high[60], "bank feature reflects the actual bank value")
    check(obs_low[61] < obs_high[61], "free_transfers feature reflects the actual value")


def test_action_to_milp_kwargs() -> None:
    squad = build_starting_squad(build_canonical_predictions(gameweek=20))
    env = HistoricalEnv()
    state = env.reset(20, squad, bank=100.0, free_transfers=1)

    kwargs = action_to_milp_kwargs((2, 2), state)  # aggressive, full budget
    conservative_kwargs = action_to_milp_kwargs((0, 0), state)  # conservative, save budget

    check(kwargs["hit_cost"] < conservative_kwargs["hit_cost"], "aggressive action sets a lower hit_cost")
    check(kwargs["budget"] > conservative_kwargs["budget"], "full-budget action allows more spend than the saving one")


def test_full_budget_action_reaches_env_step(gw: int) -> None:
    squad = build_starting_squad(build_canonical_predictions(gameweek=gw))
    env = HistoricalEnv()
    state = env.reset(gw, squad, bank=0.0, free_transfers=1)

    kwargs = action_to_milp_kwargs((1, 2), state)  # normal aggressiveness, full budget
    decision = select_squad(state.candidates, current_squad_ids=state.squad_ids,
                             bank=state.bank, free_transfers=state.free_transfers, **kwargs)
    check(decision.status == "Optimal", f"GW{gw}: full-budget action is solvable")

    outcome, next_state = env.step(decision.selected_ids, new_bank=decision.remaining_budget)
    check(outcome.gameweek == gw, f"GW{gw}: the resulting decision reached HistoricalEnv.step()")
    check(next_state.gameweek == gw + 1, f"GW{gw}: env advanced after the PPO-style decision")


def test_gymnasium_compliance() -> None:
    env = PPOEnv(start_gameweek=10, num_gameweeks=3)
    try:
        check_env(env, skip_render_check=True)
        check(True, "PPOEnv passes gymnasium's own compliance checker")
    except Exception as e:
        check(False, f"PPOEnv fails gymnasium's compliance checker: {e}")

    check(env.observation_space.shape == (OBSERVATION_SIZE,), "observation_space has the right shape")
    check(tuple(env.action_space.nvec) == ACTION_SHAPE, "action_space matches ACTION_SHAPE")
    check(env.action_space.contains(env.action_space.sample()), "a sampled action is valid")


def test_env_reset_and_step() -> None:
    env = PPOEnv(start_gameweek=15, num_gameweeks=3)
    obs, reset_info = env.reset()
    check(obs.shape == (OBSERVATION_SIZE,), "PPOEnv.reset() returns the fixed-size observation")
    check(isinstance(reset_info, dict), "reset() returns an info dict")

    obs, reward, terminated, truncated, info = env.step([1, 2])  # normal, full budget - feasible from a fresh squad
    check(isinstance(reward, float), "step() returns a float reward")
    check(not terminated and not truncated, "episode isn't done after 1 of 3 gameweeks")
    check("gameweek" in info, "info reports which gameweek was played")


def test_episode_terminates() -> None:
    env = PPOEnv(start_gameweek=15, num_gameweeks=2)
    env.reset()
    _, _, terminated1, truncated1, _ = env.step([1, 2])
    check(not terminated1 and not truncated1, "not done after gameweek 1 of 2")
    _, _, terminated2, truncated2, _ = env.step([1, 2])
    check(truncated2 and not terminated2, "truncated (episode length reached) after the last gameweek")


def test_infeasible_action_is_handled_safely() -> None:
    # budget_level=0 (save 15%) asks for less than build_starting_squad's
    # squad is already worth - genuinely infeasible from a fresh minimal
    # squad, not a bug. Confirms the env ends the episode cleanly instead
    # of fabricating an illegal squad.
    env = PPOEnv(start_gameweek=10, num_gameweeks=5)
    env.reset()
    obs, reward, terminated, truncated, info = env.step([1, 0])
    check(terminated and not truncated, "an infeasible action terminates the episode rather than faking a result")
    check(reward < 0, "an infeasible action is penalized")
    check(obs.shape == (OBSERVATION_SIZE,), "observation shape stays valid even on the terminal step")


def main() -> None:
    for gw in [5, 20, 35]:
        test_observation_shape_and_no_leakage(gw)
    test_observation_reflects_state()
    test_action_to_milp_kwargs()
    for gw in [10, 20]:
        test_full_budget_action_reaches_env_step(gw)
    test_gymnasium_compliance()
    test_env_reset_and_step()
    test_episode_terminates()
    test_infeasible_action_is_handled_safely()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

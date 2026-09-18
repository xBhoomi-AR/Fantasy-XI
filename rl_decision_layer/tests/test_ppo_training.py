"""Tests for the actual PPO agent: instantiation, a tiny training smoke
test, saving/loading, and that a loaded model's action still reaches the
existing MILP correctly.

This runs a real (very small) training loop, so it's slower than the other
tests - still just a smoke test, not real training.

Run with:
    python rl_decision_layer/tests/test_ppo_training.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO

from rl_decision_layer.environment.squad import MAX_PER_CLUB, POSITION_COUNTS, SQUAD_SIZE
from rl_decision_layer.optimization.squad_milp import select_squad
from rl_decision_layer.ppo import train as train_module
from rl_decision_layer.ppo.action import action_to_milp_kwargs
from rl_decision_layer.ppo.env import PPOEnv

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_model_instantiates() -> None:
    env = PPOEnv(start_gameweek=10, num_gameweeks=2)
    model = PPO("MlpPolicy", env, device="cpu", n_steps=32, batch_size=16, policy_kwargs=dict(net_arch=[32, 32]))
    check(model.policy is not None, "PPO model instantiates with a small CPU policy")


def test_tiny_training_and_save_load() -> None:
    env = PPOEnv(start_gameweek=10, num_gameweeks=2)
    model = PPO("MlpPolicy", env, device="cpu", n_steps=32, batch_size=16,
                policy_kwargs=dict(net_arch=[32, 32]), verbose=0)
    model.learn(total_timesteps=64)
    check(True, "a tiny (64 timestep) training run completes without error")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_model"
        model.save(path)
        check(path.with_suffix(".zip").exists(), "trained model saves to disk")

        loaded = PPO.load(path)
        check(loaded is not None, "saved model loads back")

        obs, _ = env.reset()
        action, _ = loaded.predict(obs, deterministic=True)
        check(env.action_space.contains(action), "loaded model produces a valid action")

        return action


def test_loaded_model_action_reaches_milp(action) -> None:
    env = PPOEnv(start_gameweek=20, num_gameweeks=1)
    obs, _ = env.reset()

    kwargs = action_to_milp_kwargs(tuple(action), env._state)
    decision = select_squad(env._state.candidates, current_squad_ids=env._state.squad_ids,
                             bank=env._state.bank, free_transfers=env._state.free_transfers, **kwargs)

    check(decision.status in ("Optimal", "Infeasible"), "a real trained-model action produces a real MILP result")
    if decision.status == "Optimal":
        rows = env._state.candidates[env._state.candidates["player_id"].isin(decision.selected_ids)].drop_duplicates("player_id")
        check(len(decision.selected_ids) == SQUAD_SIZE, "MILP still enforces exactly 15 players")
        for position, expected in POSITION_COUNTS.items():
            check((rows["position"] == position).sum() == expected, f"MILP still enforces {position} count")
        check((rows["team_id"].value_counts() <= MAX_PER_CLUB).all(), "MILP still enforces the club limit")


def test_checkpointing_and_resume() -> None:
    """Tiny (32-timestep) smoke test for train.py's checkpoint_freq/resume
    options, redirected to a temp MODELS_DIR so it never touches the real
    ppo/models/ppo_fpl.zip."""
    with tempfile.TemporaryDirectory() as tmp:
        original_models_dir = train_module.MODELS_DIR
        train_module.MODELS_DIR = Path(tmp)
        try:
            train_module.train(timesteps=32, start_gameweek=10, num_gameweeks=2,
                                save_name="ckpt_test", checkpoint_freq=16)
            checkpoint_dir = Path(tmp) / "checkpoints"
            checkpoints = list(checkpoint_dir.glob("ckpt_test_*.zip")) if checkpoint_dir.exists() else []
            check(len(checkpoints) > 0, "checkpoint_freq produces at least one checkpoint file")

            train_module.train(timesteps=16, start_gameweek=10, num_gameweeks=2,
                                save_name="ckpt_test", resume=True)
            check(True, "resume=True continues training from the saved model without error")
        finally:
            train_module.MODELS_DIR = original_models_dir


def main() -> None:
    test_model_instantiates()
    action = test_tiny_training_and_save_load()
    test_loaded_model_action_reaches_milp(action)
    test_checkpointing_and_resume()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

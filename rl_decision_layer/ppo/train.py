"""Trains a PPO agent on the historical FPL environment.

Run with:
    python -m rl_decision_layer.ppo.train --timesteps 500

Defaults are deliberately tiny and CPU-only - meant for you to run manually
and scale up yourself, not something Claude should run for real training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from .env import PPOEnv

MODELS_DIR = Path(__file__).resolve().parent / "models"


def train(timesteps=500, start_gameweek=1, num_gameweeks=3, device="cpu", save_name="ppo_fpl"):
    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=num_gameweeks)

    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        n_steps=64,
        batch_size=32,
        policy_kwargs=dict(net_arch=[32, 32]),
        verbose=1,
    )
    model.learn(total_timesteps=timesteps)

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / save_name
    model.save(path)
    print(f"saved model to {path}.zip")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--start-gameweek", type=int, default=1)
    parser.add_argument("--num-gameweeks", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-name", default="ppo_fpl")
    args = parser.parse_args()

    train(args.timesteps, args.start_gameweek, args.num_gameweeks, args.device, args.save_name)

"""Loads a saved PPO model and runs it through a short historical episode.

Run with:
    python -m rl_decision_layer.ppo.evaluate --model ppo_fpl
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from .env import PPOEnv
from .train import MODELS_DIR


def evaluate(model_name="ppo_fpl", start_gameweek=1, num_gameweeks=3):
    model = PPO.load(MODELS_DIR / model_name)
    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=num_gameweeks)

    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(num_gameweeks):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"GW{info.get('gameweek', '?')}: action={tuple(action)} reward={reward:.1f} info={info}")
        if terminated or truncated:
            break

    print(f"total reward: {total_reward:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_fpl")
    parser.add_argument("--start-gameweek", type=int, default=1)
    parser.add_argument("--num-gameweeks", type=int, default=3)
    args = parser.parse_args()

    evaluate(args.model, args.start_gameweek, args.num_gameweeks)

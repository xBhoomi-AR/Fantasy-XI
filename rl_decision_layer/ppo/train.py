"""Trains a PPO agent on the historical FPL environment.

Run with:
    python -m rl_decision_layer.ppo.train --timesteps 500

Defaults are deliberately tiny and CPU-only - meant for you to run manually
and scale up yourself, not something Claude should run for real training.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

warnings.filterwarnings("ignore", message=".*You are trying to run PPO on the GPU.*")

from .action import ACTION_SHAPE
from .env import PPOEnv
from .observation import OBSERVATION_SIZE

MODELS_DIR = Path(__file__).resolve().parent / "models"


def train(timesteps=20000, start_gameweek=1, num_gameweeks=15, device="cpu", save_name="ppo_fpl",
          checkpoint_freq=0, resume=False):
    """checkpoint_freq: if > 0, save a snapshot to ppo/models/checkpoints/ every
    that many timesteps. resume: if True, continue training save_name's existing model."""
    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=num_gameweeks)

    print("\n" + "=" * 65)
    print("PPO TRAINING CONFIGURATION & HYPERPARAMETERS")
    print("=" * 65)
    print(f" Execution Device       : {device.upper()}")
    print(f" Observation Dimension  : {OBSERVATION_SIZE} (incl. 4 chip availability flags)")
    print(f" Action Space Shape     : {ACTION_SHAPE} (Aggressiveness x Budget x PosBias x Chips)")
    print(f" Total Timesteps        : {timesteps:,}")
    print(f" Rollout Horizon        : 512 steps per rollout update")
    print(f" Mini-Batch Size        : 128")
    print(f" Episode Gameweeks      : {num_gameweeks} GWs per rollout episode")
    print(f" Policy Network Arch    : MLP [128, 128]")
    print(f" Learning Rate          : 3e-4")
    print(f" Entropy Coef           : 0.05 (Sustained Action Space Exploration)")
    print(f" MILP Cache Status      : {'ACTIVE (0ms step lookup)' if env.milp_cache is not None else 'DISABLED'}")
    print("=" * 65 + "\n")

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / save_name

    if resume:
        model = PPO.load(path, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            n_steps=512,
            batch_size=128,
            ent_coef=0.05,
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
        )

    callback = None
    if checkpoint_freq > 0:
        checkpoint_dir = MODELS_DIR / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=str(checkpoint_dir), name_prefix=save_name)

    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=not resume)

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
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                         help="save a checkpoint every N timesteps (0 = disabled)")
    parser.add_argument("--resume", action="store_true",
                         help="continue training --save-name's existing saved model instead of starting fresh")
    args = parser.parse_args()

    train(args.timesteps, args.start_gameweek, args.num_gameweeks, args.device, args.save_name,
          args.checkpoint_freq, args.resume)

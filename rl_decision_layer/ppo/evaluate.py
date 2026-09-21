"""Loads a saved PPO model and runs it through a short historical episode.

Run with:
    python -m rl_decision_layer.ppo.evaluate --model ppo_fpl_v4
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from .env import PPOEnv
from .train import MODELS_DIR


def evaluate(model_name="ppo_fpl_v4", start_gameweek=1, num_gameweeks=3):
    """Steps a saved model through PPOEnv (same call chain training used - no
    separate selection logic) and prints per-gameweek detail plus a summary.
    Returns a small results dict for programmatic reuse (e.g. comparisons)."""
    model = PPO.load(MODELS_DIR / model_name)
    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=num_gameweeks)

    obs, _ = env.reset()
    total_reward = 0.0
    gameweek_rewards = []
    infeasible_count = 0
    end_reason = "did not terminate/truncate"

    for _ in range(num_gameweeks):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        gameweek_rewards.append((info.get("gameweek"), reward))

        if terminated and info.get("status") and info.get("status") != "Optimal":
            infeasible_count += 1
            print(f"GW{info.get('gameweek', '?')}: action={tuple(action)} -> INFEASIBLE "
                  f"(status={info.get('status')}) reward={reward:.1f}")
        else:
            print(
                f"GW{info.get('gameweek', '?')}: action={tuple(action)} reward={reward:.1f} "
                f"squad_size={info.get('squad_size')} legal={info.get('legality_violations') == []} "
                f"xi_size={len(info.get('starting_ids', []))} "
                f"captain={info.get('captain_id')} vice={info.get('vice_captain_id')} "
                f"transfers={info.get('transfers')} hits={info.get('hits')} "
                f"bank={info.get('bank')} free_transfers={info.get('free_transfers')}"
            )

        if terminated:
            end_reason = f"terminated ({info.get('status', 'unknown')})"
            break
        if truncated:
            end_reason = "truncated (reached num_gameweeks horizon)"
            break

    print(f"\ngameweeks completed: {len(gameweek_rewards)}/{num_gameweeks}")
    print(f"infeasible decisions: {infeasible_count}")
    print(f"episode end reason: {end_reason}")
    print(f"total reward: {total_reward:.1f}")

    return {
        "total_reward": total_reward,
        "gameweek_rewards": gameweek_rewards,
        "gameweeks_completed": len(gameweek_rewards),
        "infeasible_count": infeasible_count,
        "end_reason": end_reason,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_fpl_v4")
    parser.add_argument("--start-gameweek", type=int, default=1)
    parser.add_argument("--num-gameweeks", type=int, default=3)
    args = parser.parse_args()

    evaluate(args.model, args.start_gameweek, args.num_gameweeks)

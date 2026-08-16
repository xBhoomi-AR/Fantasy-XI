"""Create cumulative return, rolling Sharpe, and drawdown curves on test data."""

import os
from pathlib import Path

# File-only backend: evaluation works on machines without a desktop Tk backend.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import (EPISODE_LENGTH, FIGURE_PATH, INITIAL_CAPITAL, MODEL_PATH,
                    SEED, STOCKS, TRANSACTION_COST, WINDOW)
from data import load_feature_data, split_data
from portfolio_env import PortfolioEnv


def trace(policy):
    _, _, test_data = split_data(load_feature_data())
    env = PortfolioEnv(test_data, STOCKS, WINDOW, INITIAL_CAPITAL, TRANSACTION_COST, EPISODE_LENGTH)
    observation, _ = env.reset(seed=SEED)
    dates, values = [env.trading_dates[env.current_day]], [env.portfolio_value]
    terminated = False
    while not terminated:
        observation, _, terminated, _, _ = env.step(policy(observation, env))
        dates.append(env.trading_dates[env.current_day])
        values.append(env.portfolio_value)
    values = np.asarray(values, dtype=float)
    daily_returns = pd.Series(values).pct_change().fillna(0.0)
    cumulative = (values / INITIAL_CAPITAL - 1.0) * 100
    sharpe = (daily_returns.rolling(20).mean() / (daily_returns.rolling(20).std() + 1e-8) * np.sqrt(252))
    drawdown = (values / np.maximum.accumulate(values) - 1.0) * 100
    return dates, cumulative, sharpe, drawdown


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    strategies = {
        "PPO": lambda observation, _: model.predict(observation, deterministic=True)[0],
        "Equal weight": lambda _, env: np.ones(env.n_stocks, dtype=np.float32) / env.n_stocks,
        "Random": lambda _, env: env.action_space.sample(),
    }
    curves = {name: trace(policy) for name, policy in strategies.items()}
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    for name, (dates, cumulative, sharpe, drawdown) in curves.items():
        axes[0].plot(dates, cumulative, label=name)
        axes[1].plot(dates, sharpe, label=name)
        axes[2].plot(dates, drawdown, label=name)
    axes[0].set(title="Cumulative Return", ylabel="Return (%)")
    axes[1].set(title="20-Day Rolling Sharpe", ylabel="Sharpe ratio")
    axes[2].set(title="Drawdown", xlabel="Date", ylabel="Drawdown (%)")
    for axis in axes:
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()

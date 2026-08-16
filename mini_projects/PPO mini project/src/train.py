"""Train PPO and save the fitted policy under results/models."""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (EPISODE_LENGTH, INITIAL_CAPITAL, MODEL_PATH, NUMBER_OF_ENVIRONMENTS,
                    SEED, STOCKS, TOTAL_TIMESTEPS, TRANSACTION_COST, WINDOW)
from data import load_feature_data, split_data
from portfolio_env import PortfolioEnv


def main():
    np.random.seed(SEED)
    train_data, _, _ = split_data(load_feature_data())

    def make_env():
        return PortfolioEnv(train_data, STOCKS, WINDOW, INITIAL_CAPITAL, TRANSACTION_COST, EPISODE_LENGTH)

    env = DummyVecEnv([make_env for _ in range(NUMBER_OF_ENVIRONMENTS)])
    model = PPO("MlpPolicy", env, device="cpu", n_steps=256, batch_size=256,
                n_epochs=5, seed=SEED, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

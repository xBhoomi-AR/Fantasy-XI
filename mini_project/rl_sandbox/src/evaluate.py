from fpl_environment import FPLEnvironment
from dqn_agent import DQNAgent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


NUM_EPISODES = 30


os.makedirs("rl_sandbox/results", exist_ok=True)


env = FPLEnvironment(seed=42)

state_size = len(env.get_state())
action_size = len(env.actions)


agent = DQNAgent(
    state_size,
    action_size
)

agent.load_model(RESULTS_DIR / "dqn_model.pth")

agent.epsilon = 0.0


def evaluate_dqn():

    rewards = []

    sample_actions = []

    for episode in range(NUM_EPISODES):

        state = env.reset(seed=42)

        done = False

        total_reward = 0

        while not done:

            action_mask = env.get_action_mask()

            action = agent.choose_action(
                state,
                action_mask
            )

            next_state, reward, done, info = env.step(action)

            if episode == 0:
                sample_actions.append(
                    env.describe_action(action)
                )

            total_reward += reward

            state = next_state

        rewards.append(total_reward)

    return np.mean(rewards), sample_actions


def evaluate_random():

    rewards = []

    for episode in range(NUM_EPISODES):

        state = env.reset(seed=42)

        done = False

        total_reward = 0

        while not done:

            valid_actions = env.get_valid_actions()

            action = random.choice(valid_actions)

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            state = next_state

        rewards.append(total_reward)

    return np.mean(rewards)


print("\nEvaluating DQN...\n")

dqn_reward, sample_actions = evaluate_dqn()

print("Evaluating Random...\n")

random_reward = evaluate_random()


print("\nRESULTS")
print("-" * 40)

print(f"Average DQN Reward    : {dqn_reward:.2f}")
print(f"Average Random Reward : {random_reward:.2f}")

print("\nSample Decisions")

for i, action in enumerate(sample_actions, start=1):

    print(f"GW{i}: {action}")


results = pd.DataFrame({

    "Agent": [
        "Random",
        "DQN"
    ],

    "Average Reward": [
        random_reward,
        dqn_reward
    ]
})

results.to_csv(
    RESULTS_DIR / "evaluation_results.csv",
    index=False
)


plt.figure(figsize=(6,5))

plt.bar(
    results["Agent"],
    results["Average Reward"]
)

plt.title("Average Reward Comparison")

plt.ylabel("Average Reward")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "comparison.png"
)

plt.close()


print("\nEvaluation completed.")
print("Results saved.")
print("Comparison graph saved.")
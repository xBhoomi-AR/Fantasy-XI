from fpl_environment import FPLEnvironment
from dqn_agent import DQNAgent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


NUM_EPISODES = 1000


# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)


env = FPLEnvironment(seed=42)

state_size = len(env.get_state())
action_size = len(env.actions)

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size
)


episode_rewards = []
episode_losses = []
episode_epsilons = []

start_time = time.time()
print("\nTraining started...\n")


for episode in range(NUM_EPISODES):

    state = env.reset(seed=42)

    done = False

    total_reward = 0

    losses = []

    while not done:

        action_mask = env.get_action_mask()

        action = agent.choose_action(
            state,
            action_mask
        )

        next_state, reward, done, info = env.step(action)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        loss = agent.learn()

        if loss is not None:
            losses.append(loss)

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    if len(losses) > 0:
        episode_losses.append(np.mean(losses))
    else:
        episode_losses.append(0)

    episode_epsilons.append(agent.epsilon)

    if (episode + 1) % 10 == 0:

        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        print(f"Reward : {episode_rewards[-1]:.2f}")
        print(f"Average Reward (Last 10): {np.mean(episode_rewards[-10:]):.2f}")
        print(f"Epsilon : {agent.epsilon:.3f}")

        if len(losses) > 0:
            print(f"Loss : {np.mean(losses):.4f}")

        print("-" * 40)


# Save model
agent.save_model("rl_sandbox/results/dqn_model.pth")


# Save training log
training_log = pd.DataFrame({
    "Episode": np.arange(1, NUM_EPISODES + 1),
    "Reward": episode_rewards,
    "Loss": episode_losses,
    "Epsilon": episode_epsilons
})

training_log.to_csv(
    "rl_sandbox/results/training_log.csv",
    index=False
)


# Reward graph
plt.figure(figsize=(8,5))
plt.plot(training_log["Episode"], training_log["Reward"])
plt.title("Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("rl_sandbox/results/reward_curve.png")
plt.close()


# Loss graph
plt.figure(figsize=(8,5))
plt.plot(training_log["Episode"], training_log["Loss"])
plt.title("Loss vs Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("rl_sandbox/results/loss_curve.png")
plt.close()


print("\nTraining completed.")
print("Model saved to rl_sandbox/results/dqn_model.pth")
print("Training log saved.")
print("Reward graph saved.")
print("Loss graph saved.")

end_time = time.time()

training_time = end_time - start_time

print(f"\nTotal Training Time : {training_time:.2f} seconds")
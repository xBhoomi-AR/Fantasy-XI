import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.model(state)


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99

        self.learning_rate = 0.001

        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998

        self.target_update_frequency = 25

        self.memory = ReplayBuffer(capacity=10000)

        self.policy_network = DQNNetwork(
            state_size,
            action_size
        ).to(DEVICE)

        self.target_network = DQNNetwork(
            state_size,
            action_size
        ).to(DEVICE)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.MSELoss()

        self.training_steps = 0

    def choose_action(self, state, action_mask):

        # Explore
        if random.random() < self.epsilon:

            valid_actions = np.where(action_mask == 1)[0]

            return random.choice(valid_actions)

        # Exploit
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = self.policy_network(state).cpu().numpy()[0]

        # Ignore invalid actions
        q_values[action_mask == 0] = -1e9

        return int(np.argmax(q_values))

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def learn(self):

        # Wait until enough experience has been collected
        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # Q values predicted by the policy network
        current_q = self.policy_network(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():

            max_next_q = self.target_network(next_states).max(dim=1)[0].unsqueeze(1)

            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_function(current_q, target_q)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.training_steps += 1

        # Update the target network every few steps
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()

        # Reduce exploration slowly
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()

    def save_model(self, path):
        torch.save(
            self.policy_network.state_dict(),
            path
        )

    def load_model(self, path):

        self.policy_network.load_state_dict(
            torch.load(path, map_location=DEVICE)
        )

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
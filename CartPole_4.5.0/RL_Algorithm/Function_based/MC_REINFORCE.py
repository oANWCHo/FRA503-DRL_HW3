from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt

class MC_REINFORCE_network(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.output = nn.Linear(hidden_size, n_actions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = self.dropout(x)

        # Stabilize softmax
        x = x - x.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(x, dim=-1)

        # Fallback check
        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum().item() == 0:
            probs = torch.ones_like(probs) / probs.size(-1)

        return probs



class MC_REINFORCE(BaseAlgorithm):
    def __init__(self, num_of_action, action_range, learning_rate, discount_factor,
                 n_observations=4, hidden_dim=256, dropout=0.1,
                 buffer_size=None, batch_size=None,
                 initial_epsilon=None, epsilon_decay=None, final_epsilon=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)
        self.device = device
        self.steps_done = 0
        self.episode_durations = []

        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()


    def calculate_stepwise_returns(self, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        else:
            print("[Warning] Only one return, skipping normalization")
        return returns
    
    

    def generate_trajectory(self, env):
        state, _ = env.reset()
        state = torch.tensor(state["policy"][0], dtype=torch.float32, device=self.device).unsqueeze(0)

        rewards = []
        log_probs = []
        trajectory = []
        ep_return = 0.0
        done = False

        while not done:
            probs = self.policy_net(state).squeeze(0)

            if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum().item() == 0:
                print("[Warning] Invalid probs detected, using uniform fallback.")
                probs = torch.ones_like(probs) / probs.size(0)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            scaled_action = self.scale_action(action.item())
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            next_state = torch.tensor(next_state["policy"][0], dtype=torch.float32, device=self.device).unsqueeze(0)
            done = terminated or truncated

            rewards.append(float(reward.item()))
            log_probs.append(log_prob)
            ep_return += reward.item()
            trajectory.append((state, action, reward))
            state = next_state

        log_prob_actions = torch.stack(log_probs).requires_grad_()
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        return ep_return, stepwise_returns, log_prob_actions, trajectory

    def calculate_loss(self, stepwise_returns, log_prob_actions):
        return -torch.sum(log_prob_actions * stepwise_returns)

    def update_policy(self, stepwise_returns, log_prob_actions):
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def learn(self, env):
        self.steps_done += 1
        self.policy_net.train()
        ep_return, returns, log_probs, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(returns, log_probs)
        return ep_return, loss, trajectory

    def select_action(self, state):
        with torch.no_grad():
            state = state["policy"][0].clone().detach().to(torch.float32).to(self.device).unsqueeze(0)
            probs = self.policy_net(state).squeeze(0)

            if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum().item() == 0:
                probs = torch.ones_like(probs) / probs.size(0)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action



    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #
    
    # def select_action(self, state):
    #     with torch.no_grad():
    #         state = self._unwrap_obs(state)
    #         state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    #         probs = self.policy_net(state)
    #         dist = torch.distributions.Categorical(probs)
    #         action = dist.sample()
    #     return action
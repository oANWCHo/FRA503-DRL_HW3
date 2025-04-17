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
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Clamp logits to prevent NaN in softmax
        x = torch.clamp(x, -20, 20)

        # Apply softmax safely
        probs = F.softmax(x, dim=-1)

        # Replace NaN with uniform fallback
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]

        return probs
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            learning_rate: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        # pass
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    def _unwrap_obs(self, obs):
        """
        Converts the observation into a flat numpy array for linear Q-learning.
        Supports dict-style obs from Isaac Sim and Gymnasium.
        """
        if isinstance(obs, dict):
            obs = obs.get("policy", obs)

        # If it's a Tensor (on CUDA or CPU)
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        # If it's a list/tuple of tensors (just in case)
        if isinstance(obs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in obs):
            obs = np.concatenate([x.detach().cpu().numpy() for x in obs])
        return np.array(obs, dtype=np.float32)  
        
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns.
        """
        # ========= put your code here ========= #
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize
        return returns
        # ====================================== #

    # def generate_trajectory(self, env):
    #     """
    #     Generate a trajectory by interacting with the environment.

    #     Args:
    #         env: The environment object.
        
    #     Returns:
    #         Tuple: (episode_return, stepwise_returns, log_prob_actions, trajectory)
    #     """
    #     # ===== Initialize trajectory collection variables ===== #
    #     # Reset environment to get initial state (tensor)
    #     # Store state-action-reward history (list)
    #     # Store log probabilities of actions (list)
    #     # Store rewards at each step (list)
    #     # Track total episode return (float)
    #     # Flag to indicate episode termination (boolean)
    #     # Step counter (int)
    #     # ========= put your code here ========= #
    #     state, _ = env.reset()
    #     state = self._unwrap_obs(state)
    #     state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    #     rewards = []
    #     log_probs = []
    #     trajectory = []
    #     ep_return = 0.0
    #     done = False
    #     timestep = 0

    #     while not done:
    #         probs = self.policy_net(state)  # Output: [1, num_actions], requires grad
    #         dist = torch.distributions.Categorical(probs)
    #         action = dist.sample()
    #         log_prob = dist.log_prob(action)  # shape: [1]
            
    #         log_probs.append(log_prob)

    #         action_tensor = torch.tensor([[action.item()]], dtype=torch.int64)
    #         next_state, reward, terminated, truncated, _ = env.step(action_tensor)
    #         done = terminated or truncated

    #         r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
    #         rewards.append(r)
    #         ep_return += r
    #         trajectory.append((state, action))

    #         state = self._unwrap_obs(next_state)
    #         state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    #         timestep += 1

    #     log_probs = torch.stack(log_probs)
    #     log_probs = log_probs.to(self.device)
    #     # log_probs.requires_grad_()  # Make sure gradient tracking is enabled  NINGGGGGGGGGGGGGGGGGGGGGGGGGGGG

    #     returns = self.calculate_stepwise_returns(rewards)

    #     return ep_return, returns.to(self.device), log_probs, trajectory
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple:
                - episode_return (float): Total reward in the episode
                - stepwise_returns (Tensor): Discounted normalized returns
                - log_prob_actions (Tensor): Log probabilities of actions
                - trajectory (list): List of (state, action) tuples
                - probs (Tensor): Action probability distributions for entropy regularization
        """
        state, _ = env.reset()
        state = self._unwrap_obs(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        rewards = []
        log_probs = []
        trajectory = []
        probs_list = []  # NEW: store action probs for each step
        ep_return = 0.0
        done = False
        timestep = 0

        while not done:
            probs = self.policy_net(state)  # [1, num_actions]
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)  # shape: [1]

            log_probs.append(log_prob)
            probs_list.append(probs)  # NEW: store current action probs

            action_tensor = torch.tensor([[action.item()]], dtype=torch.int64)
            next_state, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated

            r = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
            rewards.append(r)
            ep_return += r
            trajectory.append((state, action))

            state = self._unwrap_obs(next_state)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            timestep += 1

        # Stack into tensors
        log_probs = torch.stack(log_probs).to(self.device)           # shape: [T]
        probs = torch.cat(probs_list, dim=0).to(self.device)         # shape: [T, num_actions]

        # Compute discounted normalized returns
        returns = self.calculate_stepwise_returns(rewards)

        return ep_return, returns.to(self.device), log_probs, trajectory, probs

    
    # def calculate_loss(self, stepwise_returns, log_prob_actions):
    #     """
    #     Compute the loss for policy optimization.

    #     Args:
    #         stepwise_returns (Tensor): Stepwise returns for the trajectory.
    #         log_prob_actions (Tensor): Log probabilities of actions taken.
        
    #     Returns:
    #         Tensor: Computed loss.
    #     """
    #     # ========= put your code here ========= #
    #     return -(stepwise_returns * log_prob_actions).sum()
    #     # ====================================== #
    def calculate_loss(self, stepwise_returns, log_prob_actions, probs, entropy_coeff=0.01):
        """
        Compute the REINFORCE loss with optional entropy regularization.

        Args:
            stepwise_returns (Tensor): Discounted returns per step.
            log_prob_actions (Tensor): Log-probabilities of actions taken.
            probs (Tensor): Action probabilities at each step.
            entropy_coeff (float): Coefficient to encourage exploration.
        """
        # Loss term from REINFORCE
        reinforce_loss = -(stepwise_returns * log_prob_actions).sum()

        # Entropy of policy: -sum(p * log(p)) over all actions
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).sum()

        # Combine both losses
        total_loss = reinforce_loss - entropy_coeff * entropy
        return total_loss


    def update_policy(self, returns, log_prob_actions):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        loss = -(returns * log_prob_actions).sum()

        # print("loss.requires_grad =", loss.requires_grad)  # Should be True
        # print("log_prob_actions.requires_grad =", log_prob_actions.requires_grad)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        # ====================================== #
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        # episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        # loss = self.update_policy(stepwise_returns, log_prob_actions)
        episode_return, stepwise_returns, log_prob_actions, trajectory, probs = self.generate_trajectory(env)
        loss = self.calculate_loss(stepwise_returns, log_prob_actions, probs)

        return episode_return, loss, trajectory
        # ====================================== #


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
    
    def select_action(self, state):
        with torch.no_grad():
            state = self._unwrap_obs(state)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.policy_net(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action
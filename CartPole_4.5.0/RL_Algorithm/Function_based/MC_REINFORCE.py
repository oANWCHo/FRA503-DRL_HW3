from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function_approximation import BaseAlgorithm, ControlType


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
    Neural network สำหรับ REINFORCE algorithm
    ใช้ softmax เป็น output สำหรับสร้าง distribution ของ action

    Args:
        n_observations (int): จำนวน features จาก state
        hidden_size (int): จำนวน neurons ใน hidden layer
        n_actions (int): จำนวน discrete actions
        dropout (float): dropout สำหรับ regularization
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        
        # เลเยอร์ซ่อน: Linear + ReLU + Dropout
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
        # เลเยอร์ output: ให้ค่าความน่าจะเป็นของแต่ละ action
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        """
        กระจายข้อมูลผ่าน network และ return ค่า probability ของ action

        Args:
            x (Tensor): input state
        
        Returns:
            Tensor: softmax probabilities ของ actions
        """
        x = F.relu(self.fc1(x))         # เลเยอร์ซ่อน + ReLU
        x = self.dropout(x)             # Dropout
        x = self.fc2(x)                 # เลเยอร์ output
        return F.softmax(x, dim=-1)     # แปลงเป็น probability ด้วย softmax


class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
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

        pass
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
    
    def calculate_stepwise_returns(self, rewards):
        """
        คำนวณ return แบบสะสมย้อนหลัง (G_t = r_t + γ*r_{t+1} + ...)

        Args:
            rewards (list): รางวัลในแต่ละ timestep ของ episode
            
        Returns:
            Tensor: ค่า return สำหรับแต่ละ timestep (normalize แล้ว)
        """
        R = 0
        returns = []
        
        # วนย้อนกลับจากท้ายสุด → คำนวณ return สะสมแบบ discount
        for r in reversed(rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)  # ใส่ไว้ข้างหน้าตามลำดับเวลา

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize เพื่อช่วยให้ training เสถียรขึ้น
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        return returns


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
    #     pass
    #     # ====================================== #
        
    #     # ===== Collect trajectory through agent-environment interaction ===== #
    #     while not done:
            
    #         # Predict action from the policy network
    #         # ========= put your code here ========= #
    #         pass
    #         # ====================================== #

    #         # Execute action in the environment and observe next state and reward
    #         # ========= put your code here ========= #
    #         pass
    #         # ====================================== #

    #         # Store action log probability reward and trajectory history
    #         # ========= put your code here ========= #
    #         pass
    #         # ====================================== #
            
    #         # Update state

    #         timestep += 1
    #         if done:
    #             self.plot_durations(timestep)
    #             break

    #     # ===== Stack log_prob_actions &  stepwise_returns ===== #
    #     # ========= put your code here ========= #
    #     pass
    #     # ====================================== #
    def generate_trajectory(self, env):
        """
        เล่น 1 episode และเก็บข้อมูล trajectory:
        - log_prob ของ actions
        - รางวัล
        - state-action ทั้งหมด

        Returns:
            Tuple: (ผลรวมรางวัล, stepwise_returns, log_probs, trajectory)
        """
        obs_dict = env.reset()
        obs = obs_dict["policy"].squeeze()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        trajectory = []           # เก็บ (s, a, r)
        log_probs = []            # เก็บ log(action_prob)
        rewards = []              # เก็บรางวัล
        episode_return = 0.0
        done = False
        timestep = 0

        while not done:
            probs = self.policy_net(obs)                          # softmax ของ action
            dist = distributions.Categorical(probs)               # สร้าง distribution
            action = dist.sample()                                # sample action
            log_prob = dist.log_prob(action)                      # log prob ของ action

            continuous_action = self.scale_action(action.item())  # แปลง action index เป็นค่า
            next_obs_dict, reward, done, _ = env.step(continuous_action)
            next_obs = next_obs_dict["policy"].squeeze()
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

            # บันทึกทุกอย่าง
            trajectory.append((obs, action, reward))
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_return += reward

            obs = next_obs
            timestep += 1

            if done:
                self.plot_durations(timestep)
                break

        # คำนวณ return ย้อนหลังแบบ normalized
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        log_probs = torch.stack(log_probs)

        return episode_return, stepwise_returns, log_probs, trajectory

    
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
    #     pass
    #     # ====================================== #

    def calculate_loss(self, stepwise_returns, log_prob_actions):
        """
        คำนวณ policy gradient loss:
            loss = - ∑ logπ(a|s) * G_t

        Args:
            stepwise_returns (Tensor): Return ของแต่ละ timestep
            log_prob_actions (Tensor): log ของ prob ที่เลือก action

        Returns:
            Tensor: ค่าผลรวม loss
        """
        return -torch.sum(log_prob_actions * stepwise_returns)


    # def update_policy(self, stepwise_returns, log_prob_actions):
    #     """
    #     Update the policy using the calculated loss.

    #     Args:
    #         stepwise_returns (Tensor): Stepwise returns.
    #         log_prob_actions (Tensor): Log probabilities of actions taken.
        
    #     Returns:
    #         float: Loss value after the update.
    #     """
    #     # ========= put your code here ========= #
    #     pass
    #     # ====================================== #
    
    def update_policy(self, stepwise_returns, log_prob_actions):
        """
        อัปเดต policy ด้วย gradient descent

        Args:
            stepwise_returns (Tensor): Return ที่ normalize แล้ว
            log_prob_actions (Tensor): log ของ prob ที่เลือก action

        Returns:
            float: ค่าความสูญเสีย (loss)
        """
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
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
        episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
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

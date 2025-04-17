from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch
import torch.nn as nn

class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
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

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        self.state_dim = 4  # Customize if needed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w = np.zeros((self.state_dim, self.num_of_action), dtype=np.float32)
        self.training_error = []

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
       
    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_action: int,
        done: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().item()

        s = self._unwrap_obs(state)
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        s = np.array(s, dtype=np.float32).reshape(-1)  

        s_ = self._unwrap_obs(next_state)
        if isinstance(s_, torch.Tensor):
            s_ = s_.detach().cpu().numpy()
        s_ = np.array(s_, dtype=np.float32).reshape(-1)

        q_sa = np.dot(s, self.w[:, action])

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(np.dot(s_, self.w))

        td_error = target - q_sa
        self.w[:, action] += self.lr * td_error * s
        self.training_error.append(np.abs(td_error))

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_action)
        s = self._unwrap_obs(state)
        return int(np.argmax(self.q(s)))  # Inherited fr #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)ผ
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs, _ = env.reset()
        ep_ret = 0.0

        for _ in range(max_steps):
            action = self.select_action(obs)
            action_tensor = torch.tensor([[action]], dtype=torch.int64).to(self.device)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated

            self.update(obs, action, reward, next_obs, None, done)

            obs = next_obs
            ep_ret += reward
            

            if done:
                self.decay_epsilon(3000)
                break

        return float(ep_ret)
        # ====================================== #
    




    
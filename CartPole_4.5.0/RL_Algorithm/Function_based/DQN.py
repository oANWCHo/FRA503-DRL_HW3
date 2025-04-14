from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions)
        )
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input state tensor.
        
        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        return self.net(x)
        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.995,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 1000,
            batch_size: int = 32,
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
        if device is None:
            self.device = (
                torch.device("cuda") if torch.cuda.is_available() else
                torch.device("mps")  if torch.backends.mps.is_available() else
                torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        # ------------------------------------------------------------------ #
        # 1)  build policy & target networks directly on that device
        # ------------------------------------------------------------------ #
        self.policy_net = DQN_network(
            n_observations, hidden_dim, num_of_action, dropout
        ).to(self.device)

        self.target_net = DQN_network(
            n_observations, hidden_dim, num_of_action, dropout
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())   # hard copy
        self.target_net.eval()                                          # target net in eval‑mode

        # ------------------------------------------------------------------ #
        # 2)  optimizers & loss
        # ------------------------------------------------------------------ #
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()      # Huber loss

        # ------------------------------------------------------------------ #
        # 3)  other agent‑specific bookkeeping
        # ------------------------------------------------------------------ #
        self.batch_size = batch_size 
        self.tau = tau
        self.num_of_action = num_of_action
        self.steps_done = 0
        self.update_counter = 0            # counts gradient steps for diagnostics
        self.episode_durations: list[int] = []
        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    @staticmethod
    def _unwrap_obs(obs):
        """
        Return a 1‑D numeric vector (NumPy array or torch.Tensor) given
        an observation that may be:
        • already a vector,
        • a dict with keys like 'policy', 'obs', 'state', etc.,
        • nested one level deeper (e.g. {'policy': {'obs': ...}}).

        The function also squeezes a leading batch dimension of size 1,
        so shapes (obs_dim,) and (1, obs_dim) are both converted to
        (obs_dim,).

        Raises
        ------
        KeyError
            If no array‑like value can be found inside a dict.
        """
        # 1) If it's already a tensor / ndarray / list, we're done.
        if isinstance(obs, (np.ndarray, list, tuple, torch.Tensor)):
            vec = obs
        # 2) If it's a dict, look for the common keys.
        elif isinstance(obs, dict):
            candidate = None
            for key in ("policy", "obs", "state", "policy_obs", "observation"):
                if key in obs:
                    candidate = obs[key]
                    break
            if candidate is None:
                raise KeyError(f"_unwrap_obs: no numeric vector in keys {list(obs.keys())}")

            # If the candidate is itself a dict, grab the first array‑like value.
            if isinstance(candidate, dict):
                for v in candidate.values():
                    if isinstance(v, (np.ndarray, list, tuple, torch.Tensor)):
                        candidate = v
                        break
                else:
                    raise KeyError("_unwrap_obs: nested dict contains no array‑like values")

            vec = candidate
        else:
            raise TypeError(f"_unwrap_obs: unsupported type {type(obs)}")

        # 3) Convert lists / tuples to NumPy for consistency.
        if isinstance(vec, (list, tuple)):
            vec = np.asarray(vec, dtype=np.float32)

        # 4) Remove a leading batch dim of size 1 (shape (1, obs_dim) → (obs_dim,))
        if hasattr(vec, "ndim") and vec.ndim == 2 and vec.shape[0] == 1:
            vec = vec.squeeze(0)

        return vec
    
    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if isinstance(state, dict):
            # ordered list of likely keys
            for k in ("obs", "state", "policy", "policy_obs", "observation"):
                if k in state:
                    state = state[k]
                    # if it’s still a dict (e.g. {"obs": ...}) unwrap one more level
                    if isinstance(state, dict):
                        # try to grab the first ndarray in that sub‑dict
                        for v in state.values():
                            if isinstance(v, (np.ndarray, list, tuple, torch.Tensor)):
                                state = v
                                break
                    break
            else:
                raise KeyError(
                    f"select_action: could not find observation vector in keys {list(state.keys())}"
                )

        sample = random.random()
        eps_threshold = self.epsilon

        if sample > eps_threshold:
            with torch.no_grad():
                # after you have extracted the ndarray into `state`
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

                # If state already has a batch dimension (shape [1, obs_dim]) do NOT add another
                if state_t.ndim == 1:          # shape (obs_dim,)
                    state_t = state_t.unsqueeze(0)        # → (1, obs_dim)

                # forward pass
                q_vals = self.policy_net(state_t)         # shape (1, num_actions)

                action = q_vals.argmax(dim=1).item() 
        else:
            action = random.randrange(self.num_of_action)
        return action
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken.
            reward_batch (Tensor): Batch of received rewards.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        # Current Q(s,a)
        q_sa = self.policy_net(state_batch).gather(1, action_batch)

        # Target Q
        next_q = torch.zeros_like(reward_batch).to(self.device)
        with torch.no_grad():
            next_q_vals = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
            next_q[non_final_mask] = next_q_vals
        target_q = reward_batch + self.discount_factor * next_q

        return self.criterion(q_sa, target_q)
        # ====================================== #

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor): A boolean mask indicating which states are non-final.
                - non_final_next_states (Tensor): The next states that are not terminal.
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        # batch = self.memory.sample()
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample()
        # unwrap dict observations
        states      = [ self._unwrap_obs(s)  for s in states ]
        next_states = [ self._unwrap_obs(s)  for s in next_states ]
        state_batch = torch.stack(
            [torch.as_tensor(s, dtype=torch.float32, device=self.device).view(-1) for s in states]
        )  
        action_batch = torch.tensor(actions, dtype=torch.int64,   device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        # mask for non‑terminal
        non_final_mask = torch.tensor(
            [not d for d in dones], dtype=torch.bool, device=self.device
        )
        non_final_next_states = torch.stack(
            [torch.as_tensor(s, dtype=torch.float32, device=self.device).view(-1)
            for s, d in zip(next_states, dones) if not d]
        ) if non_final_mask.any() else torch.empty((0, state_batch.size(1)),
                                                dtype=torch.float32, device=self.device)

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # Generate a sample batch
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        
        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # Perform gradient descent step
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        policy_state  = self.policy_net.state_dict()   # online / policy
        target_state  = self.target_net.state_dict()   # target (to be updated)
        # ====================================== #
        
        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        tau = self.tau
        for key in policy_state:                       # same keys in both dicts
            target_state[key] = tau * policy_state[key] + (1.0 - tau) * target_state[key]
        # ====================================== #
        
        # Load the updated weights into the target network
        # ========= put your code here ========= #
        self.target_net.load_state_dict(target_state)
        # ====================================== #

    def learn(self, env):
        """
        Train the agent on a single step.

        Args:
            env: The environment to train in.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        state, _ = env.reset()           # state : np.ndarray
        total_reward = 0.0
        done = False
        timestep = 0
        # ====================================== #

        while not done:
            # Predict action from the policy network
            # ========= put your code here ========= #
            action = self.select_action(state)          # int ∈ [0, num_actions)
            action_tensor = torch.tensor([[action]], dtype=torch.int64)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_state, reward, terminated, truncated,_ = env.step(action_tensor)
            done = terminated or truncated
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            self.memory.add(state, action, reward, next_state, done)

            total_reward += float(reward.item())
            # ====================================== #

            # Update state

            # Perform one step of the optimization (on the policy network)
            self.update_policy()

            # Soft update of the target network's weights
            self.update_target_networks()

            state = next_state
            timestep += 1
            if done:
                self.plot_durations(timestep)
                self.decay_epsilon()
                return total_reward

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
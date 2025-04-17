import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # ========= put your code here ========= #
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Use a non-linear activation for hidden layers
        self.relu = nn.ReLU()
        # For discrete action space, we apply Softmax across actions
        self.tanh = nn.Tanh()

        # Define an optimizer for this network
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize weights
        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        # ========= put your code here ========= #
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        return x
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for Q-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        # ========= put your code here ========= #
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Outputs a single Q-value
      
        self.relu = nn.ReLU()

        # Define the optimizer for this critic network
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize weights
        self.init_weights()  
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state, action):
        """
        Forward pass for Q-value estimation.

        Args:
            state (Tensor): Current state of the environment.
            action (Tensor): Action taken by the agent.

        Returns:
            Tensor: Estimated Q-value.
        """
        # ========= put your code here ========= #
        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], dim=-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value
        # ====================================== #

class Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #

        # self.device = torch.device(device) if device is not None else torch.device("cpu")
        # self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        # self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        # self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        # self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        # self.batch_size = batch_size
        # self.tau = tau
        # self.discount_factor = discount_factor

        # self.update_target_networks(tau=1)  # initialize target networks

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        # pass
        # ====================================== #

        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        #========================================================= #
        self.device = (torch.device(device) if device is not None else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # ---- networks ----------------------------------------------------
        self.actor         = Actor(n_observations, hidden_dim, num_of_action).to(self.device)
        self.actor_target  = Actor(n_observations, hidden_dim, num_of_action).to(self.device)
        self.critic        = Critic(n_observations, num_of_action, hidden_dim).to(self.device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ---- optimizers --------------------------------------------------
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # ---- misc --------------------------------------------------------
        self.tau     = tau
        self.gamma   = discount_factor
        self.batch_size = batch_size
        self.mse     = nn.MSELoss()
        # ========================================================= #

        # Additional parameters
        # self.batch_size = batch_size
        # self.tau = tau
        # self.discount_factor = discount_factor

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
        """Return flat numeric vector from possibly nested dict."""
        if isinstance(obs, dict):
            for k in ("policy", "obs", "state", "observation"):
                if k in obs:
                    return Actor_Critic._unwrap_obs(obs[k])
            raise KeyError("No array in observation dict")
        if isinstance(obs, (list, tuple)):
            obs = np.asarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.ndim == 2 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        return obs

    def scale_action(self, action_vec):
        """
        Parameters
        ----------
        action_vec : np.ndarray | torch.Tensor | list
            1‑D vector whose components are in the range (-1, 1).

        Returns
        -------
        torch.Tensor  shape = [1, act_dim]   (same device as the agent)
            Scaled actions in the user‑defined action_range.
        """
        if isinstance(action_vec, (list, tuple)):
            action_vec = np.asarray(action_vec, dtype=np.float32)
        if isinstance(action_vec, np.ndarray):
            action_vec = torch.from_numpy(action_vec)
        action_vec = action_vec.to(self.device).float()        # 1‑D tensor

        a_min, a_max = self.action_range
        scaled = a_min + 0.5 * (action_vec + 1.0) * (a_max - a_min)  # element‑wise
        return scaled.unsqueeze(0)    
    
    def select_action(self, state, noise=0.0):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        # ========= put your code here ========= #
        # Convert state to torch if it's a NumPy array
        s = self._unwrap_obs(state).to(self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)                         # (1, obs_dim)

        with torch.no_grad():
            a = self.actor(s)[0].cpu().numpy()         # in (‑1,1)

        if noise > 0.0:
            a += noise * np.random.randn(*a.shape)

        a = np.clip(a, -1.0, 1.0)
        return self.scale_action(a), a   
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Sample a batch from replay buffer and convert to tensors.
        Handles multi-agent reward shapes correctly.
        """
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Unwrap and convert observations to tensor
        state_batch  = torch.stack([self._unwrap_obs(s) for s in states]).to(self.device)
        action_batch = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in actions]).to(self.device)
        next_state_batch = torch.stack([self._unwrap_obs(s2) for s2 in next_states]).to(self.device)

        # Convert rewards/dones safely
        reward_batch = torch.stack([r if torch.is_tensor(r) else torch.tensor(r, dtype=torch.float32)
                            for r in rewards]).to(self.device).view(-1, 1)
        done_batch   = torch.stack([d if torch.is_tensor(d) else torch.tensor(d, dtype=torch.float32)
                          for d in dones]).to(self.device).view(-1, 1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        
        # ----- Critic Update -----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_tgt = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)

        q      = self.critic(states, actions)
        critic = self.mse(q, q_tgt)

        actor  = -self.critic(states, self.actor(states)).mean()
        return critic.item(), actor.item()
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        """One gradient step for critic then actor (no in‑place clash)."""
        batch = self.generate_sample(self.batch_size)
        if batch is None:
            return None,None

        states, actions, rewards, next_states, dones = batch
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # ---------- Critic update ------------------------------------------
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = rewards + self.gamma * (1 - dones) * \
                    self.critic_target(next_states, next_actions)

        q     = self.critic(states, actions)
        critic_loss = self.mse(q, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # ---------- Actor update  (re‑compute with fresh critic) ------------
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # ---------- Soft update of targets ---------------------------------
        with torch.no_grad():
            for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)
            for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)

        return actor_loss.item(),critic_loss.item()
        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        if tau is None:
            tau = self.tau

        # Update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # ====================================== #

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        state, _ = env.reset()

        if isinstance(state, dict) and 'policy' in state:
            state = state['policy']

        ep_ret = 0.0
        noise = noise_scale
        last_a_loss = None
        last_c_loss = None

        for _ in range(max_steps):
            act_scaled_list, act_raw_list = [], []

            for i in range(num_agents):
                a_s, a_r = self.select_action(state[i], noise)
                act_scaled_list.append(a_s)
                act_raw_list.append(a_r)

            act_scaled = torch.cat(act_scaled_list, dim=0)  # shape [num_agents, action_dim]
            step_out = env.step(act_scaled)

            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                if isinstance(next_state, dict) and 'policy' in next_state:
                    next_state = next_state['policy']
                done_flag = [t or tr for t, tr in zip(terminated, truncated)]
            else:
                next_state, reward, done_flag, _ = step_out

            for i in range(num_agents):
                r = reward[i]
                d = done_flag[i]
                r = r.item() if torch.is_tensor(r) and r.numel() == 1 else float(r[0]) if torch.is_tensor(r) else float(r)
                d = d.item() if torch.is_tensor(d) and d.numel() == 1 else bool(d[0]) if torch.is_tensor(d) else bool(d)

                self.memory.add(
                    self._unwrap_obs(state[i]),
                    torch.as_tensor(act_raw_list[i], dtype=torch.float32),
                    r,
                    self._unwrap_obs(next_state[i]),
                    d,
                )
                ep_ret += r

            a_loss, c_loss = self.update_policy()
            if a_loss is not None and c_loss is not None:
                last_a_loss = a_loss
                last_c_loss = c_loss
            state = next_state
            noise *= noise_decay

            if all(bool(d) for d in done_flag):
                break


        return float(ep_ret), last_a_loss, last_c_loss

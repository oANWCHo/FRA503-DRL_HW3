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
        self.softmax = nn.Softmax(dim=-1)

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
        x = self.softmax(x)
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
                dropout = 0.05, 
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
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor

        self.update_target_networks(tau=1)  # initialize target networks

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
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
        self.device = device if device is not None else "cpu"
        
        # Initialize actor and critic (online networks)
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(self.device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(self.device)

        # Initialize target networks
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(self.device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(self.device)

        # Copy weights from online to target (Polyak update with tau=1)
        self.update_target_networks(tau=1.0)
        # ========================================================= #

        # Additional parameters
        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor

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
        if not isinstance(state, torch.Tensor):
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # shape [1, state_dim]
        else:
            state_t = state.to(self.device).unsqueeze(0)

        # Put actor in evaluation mode if you want to avoid dropout, etc.
        self.actor.eval()
        with torch.no_grad():
            # Actor outputs an unbounded or [-1,1]-bounded action (depends on your Actor's design)
            action = self.actor(state_t)
        self.actor.train()

        # Convert to NumPy for noise processing
        action = action.cpu().numpy()[0]  # shape: [action_dim]

        # Add exploration noise (for continuous control)
        if noise > 0.0:
            action += noise * np.random.randn(*action.shape)

        # Clip action to [-1, 1] if your actor is designed that way
        clipped_action = np.clip(action, -1.0, 1.0)

        # Scale action from [-1,1] to [action_min, action_max]
        scaled_action = self.scale_action(clipped_action)

        return scaled_action, clipped_action
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        batch = self.memory.sample()
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Convert to torch tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)  # shape [batch_size, 1]
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)      # shape [batch_size, 1]
        
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
        # 1) Compute target Q-value
        with torch.no_grad():
            # Actor target to get next action
            next_actions = self.actor_target(next_states)
            # Critic target to get Q(s', a')
            target_Q = self.critic_target(next_states, next_actions)
            # Bellman backup
            y = rewards + self.discount_factor * (1 - dones) * target_Q

        # 2) Current Q estimate
        current_Q = self.critic(states, actions)

        # 3) Critic loss = MSE of (current_Q - y)
        critic_loss_fn = nn.MSELoss()
        critic_loss = critic_loss_fn(current_Q, y)

        # ----- Actor Update -----
        # Policy gradient update: maximize Q(s, a) w.r.t. the actor’s parameters
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        return critic_loss, actor_loss
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return  # Not enough samples yet

        states, actions, rewards, next_states, dones = sample

        # (Optional) Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Calculate losses
        critic_loss, actor_loss = self.calculate_loss(states, actions, rewards, next_states, dones)

        # --- Update Critic ---
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        # Optional gradient clipping
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic.optimizer.step()

        # --- Update Actor ---
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # Optional gradient clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor.optimizer.step()
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
        state = env.reset()
        total_reward = 0.0
        noise = noise_scale
        # ====================================== #

        for step in range(max_steps):
            # Predict action from the policy network
            # ========= put your code here ========= #
            if num_agents > 1:
                # Multi-Agent scenario
                scaled_actions, clipped_actions = [], []
                for agent_i in range(num_agents):
                    s_i = state[agent_i]
                    a_scaled, a_clipped = self.select_action(s_i, noise=noise)
                    scaled_actions.append(a_scaled)
                    clipped_actions.append(a_clipped)
            else:
                # Single-Agent scenario
                scaled_action, clipped_action = self.select_action(state, noise=noise)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            if num_agents > 1:
                next_state, reward, done, info = env.step(scaled_actions)
            else:
                next_state, reward, done, info = env.step(scaled_action)
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            # Parallel Agents Training
            if num_agents > 1:
            # Parallel Agents Training
                for agent_i in range(num_agents):
                    self.memory.add(
                        state[agent_i],
                        clipped_actions[agent_i],
                        reward[agent_i],
                        next_state[agent_i],
                        done[agent_i]
                    )
                total_reward += sum(reward)
            # Single Agent Training
            else:
                self.memory.add(state, clipped_action, reward, next_state, done)
                total_reward += reward
            # ====================================== #

            # Update state
            if num_agents > 1:
                state = next_state
                # If all agents are done, break
                if all(done):
                    break
            else:
                state = next_state
                if done:
                    break

            # Decay the noise to gradually shift from exploration to exploitation
            noise *= noise_decay

            # Perform one step of the optimization (on the policy network)
            self.update_policy()

            # Update target networks
            self.update_target_networks()

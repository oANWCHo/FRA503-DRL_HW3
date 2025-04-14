# from __future__ import annotations
# import numpy as np
# from RL_Algorithm.RL_base_function import BaseAlgorithm


# class Linear_QN(BaseAlgorithm):
#     def __init__(
#             self,
#             num_of_action: int = 2,
#             action_range: list = [-2.5, 2.5],
#             learning_rate: float = 0.01,
#             initial_epsilon: float = 1.0,
#             epsilon_decay: float = 1e-3,
#             final_epsilon: float = 0.001,
#             discount_factor: float = 0.95,
#     ) -> None:
#         """
#         Initialize the CartPole Agent.

#         Args:
#             learning_rate (float): The learning rate for updating Q-values.
#             initial_epsilon (float): The initial exploration rate.
#             epsilon_decay (float): The rate at which epsilon decays over time.
#             final_epsilon (float): The final exploration rate.
#             discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
#         """        

#         super().__init__(
#             num_of_action=num_of_action,
#             action_range=action_range,
#             learning_rate=learning_rate,
#             initial_epsilon=initial_epsilon,
#             epsilon_decay=epsilon_decay,
#             final_epsilon=final_epsilon,
#             discount_factor=discount_factor,
#         )
        
#     def update(
#         self,
#         obs,
#         action: int,
#         reward: float,
#         next_obs,
#         next_action: int,
#         terminated: bool
#     ):
#         """
#         Updates the weight vector using the Temporal Difference (TD) error 
#         in Q-learning with linear function approximation.

#         Args:
#             obs (dict): The current state observation, containing feature representations.
#             action (int): The action taken in the current state.
#             reward (float): The reward received for taking the action.
#             next_obs (dict): The next state observation.
#             next_action (int): The action taken in the next state (used in SARSA).
#             terminated (bool): Whether the episode has ended.

#         """
#         # ========= put your code here ========= #
#         pass
#         # ====================================== #

#     def select_action(self, state):
#         """
#         Select an action based on an epsilon-greedy policy.
        
#         Args:
#             state (Tensor): The current state of the environment.
        
#         Returns:
#             Tensor: The selected action.
#         """
#         # ========= put your code here ========= #
#         pass
#         # ====================================== #

#     def learn(self, env, max_steps):
#         """
#         Train the agent on a single step.

#         Args:
#             env: The environment in which the agent interacts.
#             max_steps (int): Maximum number of steps per episode.
#         """

#         # ===== Initialize trajectory collection variables ===== #
#         # Reset environment to get initial state (tensor)
#         # Track total episode return (float)
#         # Flag to indicate episode termination (boolean)
#         # Step counter (int)
#         # ========= put your code here ========= #
#         pass
#         # ====================================== #
    


from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
            buffer_size: int = 1000,
            batch_size: int = 1,
    ) -> None:
        # เรียกใช้งาน constructor ของ BaseAlgorithm เพื่อกำหนดพารามิเตอร์พื้นฐาน
        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size
        )

    def update(self, obs, action: int, reward: float, next_obs, next_action: int, terminated: bool):
        """
        ฟังก์ชันอัปเดตน้ำหนักของ Linear Q-Network ตามกฎของ Q-learning
        """

        # ✅ แปลง state จาก tensor → numpy (บน CPU)
        if isinstance(obs, torch.Tensor):
            phi_s = obs.detach().cpu().numpy()
        else:
            phi_s = np.array(obs)

        if isinstance(next_obs, torch.Tensor):
            phi_next = next_obs.detach().cpu().numpy()
        else:
            phi_next = np.array(next_obs)

        q_sa = self.q(phi_s, action)

        if terminated:
            target = reward
        else:
            q_next = self.q(phi_next)
            target = reward + self.discount_factor * np.max(q_next)

        td_error = float(target - q_sa)  # ✅ แปลงให้เป็น float ก่อนคูณ numpy
        self.w[:, action] += self.lr * td_error * phi_s
        self.training_error.append(td_error)


    def select_action(self, obs):
        """
        ฟังก์ชันเลือก action ตามนโยบาย epsilon-greedy
        มีความน่าจะเป็น epsilon ที่จะสุ่ม action เพื่อสำรวจ
        """
        if isinstance(obs, torch.Tensor):
            phi_s = obs.detach().cpu().numpy()
        else:
            phi_s = np.array(obs)


        if np.random.rand() < self.epsilon:
            # สำรวจ: สุ่มเลือก action
            action = np.random.choice(self.num_of_action)
        else:
            # ใช้ประโยชน์: เลือก action ที่มี Q สูงสุด
            q_values = self.q(phi_s)
            action = int(np.argmax(q_values))

        return action

    def learn(self, env, max_steps):
        """
        ฟังก์ชันฝึก Linear Q-Learning agent สำหรับ IsaacLab environment
        รองรับกรณีที่ env.reset() คืน (dict, info) และ dict มี key 'policy'
        """
        # ----- RESET ENVIRONMENT ----- #
        reset_result = env.reset()

        if not isinstance(reset_result, tuple):
            raise TypeError("Expected env.reset() to return a tuple (obs_dict, info)")

        obs_dict = reset_result[0]

        if not isinstance(obs_dict, dict) or "policy" not in obs_dict:
            raise TypeError("env.reset()[0] must be a dict with key 'policy'")

        obs = obs_dict["policy"].squeeze()  # ✅ ดึง state แบบ tensor 1 มิติ

        # ----- INITIALIZE VARIABLES ----- #
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            action = self.select_action(obs)
            continuous_action = self.scale_action(action)

            # ----- STEP ENVIRONMENT ----- #
            step_result = env.step(continuous_action)

            if not isinstance(step_result, tuple) or len(step_result) < 5:
                raise TypeError("env.step() must return 5 values: (obs_dict, reward, terminated, truncated, info)")

            next_obs_dict, reward, terminated, truncated, _ = step_result

            if not isinstance(next_obs_dict, dict) or "policy" not in next_obs_dict:
                raise TypeError("env.step()[0] must be a dict with key 'policy'")

            next_obs = next_obs_dict["policy"].squeeze()
            done = terminated or truncated

            # ----- LEARNING ----- #
            self.memory.add(obs, action, reward, next_obs, done)

            if len(self.memory) >= self.memory.batch_size:
                batch = self.memory.sample()
                for b_obs, b_action, b_reward, b_next_obs, b_done in zip(*batch):
                    self.update(b_obs, b_action, b_reward, b_next_obs, None, b_done)

            obs = next_obs
            total_reward += reward
            step += 1

        self.decay_epsilon()
        return total_reward


    
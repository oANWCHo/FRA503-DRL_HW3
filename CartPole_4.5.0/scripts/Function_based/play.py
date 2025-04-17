"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN
from RL_Algorithm.Function_based.Linear_Q import Linear_QN
from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
from RL_Algorithm.Function_based.AC import Actor_Critic

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters lq
    num_of_action = 5
    action_range = [-15.0, 15.0]
    learning_rate = 0.005
    initial_epsilon = 0
    epsilon_decay = 0
    final_epsilon = 0
    discount = 0.99
    n_episodes = 10000
    max_steps = 2000

    # hyperparameter dqn
    # num_of_action = 5
    # action_range = [-25.0, 25.0]
    # n_observations = 4
    # hidden_dim = 4 #4 64 128 256 
    # dropout = 0.0
    # learning_rate = 0.0005
    # tau = 0.001 
    # initial_epsilon = 0
    # epsilon_decay = 0
    # final_epsilon = 0
    # discount_factor = 0.99
    # buffer_size = 50000
    # batch_size = 512  
    # n_episodes = 10000

    # hyperparameter MC
    # num_of_action = 5 
    # action_range = [-7, 7] 
    # n_observations = 4 
    # hidden_dim = 128 
    # learning_rate = 0.005  
    # dropout = 0.0
    # discount_factor = 0.99
    # n_episodes = 10000

    # hyperparameters AC
    # num_of_action = 1
    # action_range = [-16.0, 16.0]  # [min, max]
    # n_observations = 4
    # learning_rate = 0.001 #0.0005
    # hidden_dim = 64 #64 128
    # tau = 0.005 #0.001 0.01
    # buffer_size = 5000 #5000 7000 3000 
    # batch_size = 64 #64 128 256
    # discount_factor = 0.99 #0.1
    # noise_scale_init     = 0.2
    # noise_decay          = 0.995
    # n_episodes = 10000
    # max_steps_per_episode = 1000
    # num_agents = args_cli.num_envs    


    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    agent = Linear_QN(
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount,
    )

    # agent = DQN(
    #     device=device,
    #     num_of_action=num_of_action,
    #     action_range=action_range,
    #     learning_rate=learning_rate,
    #     hidden_dim=hidden_dim,
    #     initial_epsilon = initial_epsilon,
    #     epsilon_decay = epsilon_decay,
    #     final_epsilon = final_epsilon,
    #     discount_factor = discount,
    #     buffer_size = buffer_size,
    #     batch_size = batch_size,
    # )

    # agent = MC_REINFORCE(
    #     device=device,
    #     num_of_action=num_of_action,
    #     action_range=action_range,
    #     n_observations=n_observations,
    #     hidden_dim=hidden_dim,
    #     dropout=dropout,
    #     learning_rate=learning_rate,
    #     discount_factor=discount_factor
    # )

    # agent = Actor_Critic(
    #     device = device, 
    #     num_of_action  = num_of_action,
    #     action_range = action_range,
    #     n_observations = n_observations,
    #     hidden_dim = hidden_dim,
    #     learning_rate = learning_rate,
    #     tau = tau,
    #     discount_factor = discount_factor,
    #     buffer_size = buffer_size,
    #     batch_size = batch_size,
    # )

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "DQN"  
    episode = 0
    q_value_file = f"ep{episode}_lr{learning_rate}_na{num_of_action}_acr{action_range[1]}_ms{max_steps}_dis{discount}.npy" #LQ
    # q_value_file = f"dqn_na{num_of_action}_ep{episode+1}_lr{learning_rate}_bs{batch_size}_dis{discount_factor}_τ{tau}_hd{hidden_dim}.pt" #DQN
    # q_value_file   =  f"ep{episode}_na{num_of_action}_lr{learning_rate}_hd{hidden_dim}_dp{dropout}_dis{discount_factor}.pt" #MC
    # q_value_file = f"ac_agents{num_agents}_ep{episode}_lr{learning_rate}_bs{batch_size}_dis{discount_factor}_τ{tau}_hd{hidden_dim}.pt" #AC

    full_path = os.path.join(f"w/{task_name}", Algorithm_name)
    agent.load_w(full_path, q_value_file)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False

                while not done:
                    # agent stepping
                    action, action_idx = agent.select_action(obs)
                    action_tensor = torch.tensor([[action]], dtype=torch.int64)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action_tensor)

                    done = terminated or truncated
                    obs = next_obs
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
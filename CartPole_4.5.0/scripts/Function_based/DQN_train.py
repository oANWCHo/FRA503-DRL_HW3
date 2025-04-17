"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN

from tqdm import tqdm
import torch
import wandb

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source/CartPole")))
###ใส่ 2 อันเลย
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

    # hyperparameters
    num_of_action = 5
    action_range = [-25.0, 25.0]

    n_observations = 4
    hidden_dim = 4 #4 64 128 256 
    dropout = 0.0

    learning_rate = 0.0005
    tau = 0.001 

    initial_epsilon = 1.0
    epsilon_decay = 0.9997
    final_epsilon = 0.01

    discount_factor = 0.99

    buffer_size = 50000
    batch_size = 512  

    n_episodes = 10000

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

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "DQN"

    agent = DQN(
        device=device,
        num_of_action = num_of_action,
        action_range = action_range,
        n_observations = n_observations,
        hidden_dim = hidden_dim,
        dropout = dropout,
        learning_rate = learning_rate,
        tau = tau,
        initial_epsilon = initial_epsilon,
        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor = discount_factor,
        buffer_size = buffer_size,
        batch_size = batch_size,
    )
    config = {
        'architecture': 'HW3_DRL', 
        'name' : 'DQN'
    }
    run = wandb.init(
        project='HW3_DRL',

        config=config,
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    all_rewards = []
    all_loss = []

    # Outer loop: while simulation is running
    while simulation_app.is_running():

        # for each episode
        for episode in tqdm(range(n_episodes), desc="Episode"):
            # reset environment for this new episode
            ep_reward, ep_loss = agent.learn(env)
            all_rewards.append(ep_reward)
                        
            if ep_loss is not None:
                all_loss.append(ep_loss)
            # Episode done → store total reward
            # print(ep_loss)
            wandb.log({
                        "episode_reward": ep_reward,
                        "episode_loss": ep_loss if ep_loss is not None else 0.0,
                         "epsilon": agent.epsilon
                    }, step=episode)


            # Example checkpoint logic
            if (episode + 1) % 100 == 0:

                avg_reward = np.mean(all_rewards[-100:])
                avg_loss = np.mean(all_loss[-100:])
                print(f"[Episode {episode+1:4d}]   AvgReward(Last100) = {avg_reward:.2f}   Epsilon = {agent.epsilon:.3f}")
                wandb.log({
                    "avg_reward_100": avg_reward,
                    "avg_loss_100": avg_loss
                    },  step=episode)

                ckpt_dir = os.path.join("weights", task_name, "DQN")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_filename = (
                    f"dqn_na{num_of_action}_ep{episode+1}"
                    f"_lr{learning_rate}"
                    f"_bs{batch_size}"
                    f"_dis{discount_factor}"
                    f"_τ{tau}"
                    f"_hd{hidden_dim}.pt"
                )
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

                agent.save_w(ckpt_dir, ckpt_filename)
                print(f"Checkpoint saved → {ckpt_path}")


        print("Training Complete.")
        # agent.plot_durations(show_result=True)  # if your agent uses that
        # plt.ioff()
        # plt.show()
            
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
    wandb.finish()
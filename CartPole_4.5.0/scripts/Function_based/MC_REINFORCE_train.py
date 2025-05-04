"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source/CartPole")))

# Import extensions to set up environment tasks
import CartPole.tasks # noqa: F401

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

    # Hyperparameters
    num_of_action = 21 #  if num_of_ac end please run [-2.5,2.5] num_ac 2
    action_range = [-20, 20] 
    n_observations = 4 # Fix

    hidden_dim = 64 # 32 64(base) 256 
    learning_rate = 0.001 #0.001(base)  0.01  
    dropout = 0.5 #0.5 0.3 0.7
    discount_factor = 0.99

    n_episodes = 2000

    task_name = str(args_cli.task).split('-')[0]  # เช่น Stabilize

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Agent
    agent = MC_REINFORCE(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        n_observations=n_observations,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        discount_factor=discount_factor
    )

    # wandb init
    config = {
        'architecture': 'HW3_DRL', 
        'name' : 'MC_reinforce'
    }
    run = wandb.init(
        project='HW3_DRL',

        config=config,
    )
    while simulation_app.is_running():
        all_rewards = []
        all_loss = []
        for episode in tqdm(range(n_episodes)):
            ep_return, ep_loss, _ = agent.learn(env)
            all_rewards.append(ep_return)
            if ep_loss is not None:
                all_loss.append(ep_loss)
            wandb.log({
                        "episode_reward": ep_return,
                        "episode_loss": ep_loss if ep_loss is not None else 0.0,
                    }, step=episode)
            
            if (episode + 1) % 100 == 0:
                avg_reward = float(np.mean(all_rewards[-100:]))
                avg_loss = np.mean([
                    l.detach().cpu().item() if isinstance(l, torch.Tensor) else l
                    for l in all_loss[-100:]
                ])

                print(f"[Episode {episode + 1}] AvgReward(Last100) = {avg_reward:.2f}")
                wandb.log({
                    "avg_reward_100": avg_reward,
                    "avg_loss_100": avg_loss
                    },  step=episode)


                ckpt_dir = os.path.join("weights", task_name, "MC_REINFORCE")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_filename = (
                    f"ep{episode}"
                    f"_na{num_of_action}"
                    f"_lr{learning_rate}"
                    f"_hd{hidden_dim}"
                    f"_dp{dropout}"
                    f"_dis{discount_factor}.pt"
                )
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                torch.save(agent.policy_net.state_dict(), ckpt_path)
                print(f"Checkpoint saved → {ckpt_path}")

        print("Training complete.")
        agent.plot_durations(show_result=True)
        env.close()
        wandb.finish()
        simulation_app.close()


if __name__ == "__main__":
    main()

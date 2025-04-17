"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.AC import Actor_Critic

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

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 1
    action_range = [-16.0, 16.0]  # [min, max]
    n_observations = 4
    learning_rate = 0.001 #0.0005
    hidden_dim = 64 #64 128
    tau = 0.005 #0.001 0.01
    buffer_size = 5000 #5000 7000 3000 


    batch_size = 64 #64 128 256
    discount_factor = 0.99 #0.1

    noise_scale_init     = 0.2
    noise_decay          = 0.995

    n_episodes = 10000
    max_steps_per_episode = 1000

    num_agents = args_cli.num_envs

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
    Algorithm_name = "AC"

    agent = Actor_Critic(
        device = device, 
        num_of_action  = num_of_action,
        action_range = action_range,
        n_observations = n_observations,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        tau = tau,
        discount_factor = discount_factor,
        buffer_size = buffer_size,
        batch_size = batch_size,
    )


    config = {
        'architecture': 'HW3_DRL', 
        'name' : 'AC'
    }
    run = wandb.init(
        project='HW3_DRL',

        config=config,
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0



    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        # with torch.inference_mode():
        all_rewards = []
        all_a_loss = []
        all_c_loss = []
        for episode in tqdm(range(n_episodes)):
            ep_reward,a_loss,c_loss = agent.learn(
                env=env,
                max_steps=max_steps_per_episode,
                num_agents=num_agents,
                noise_scale=noise_scale_init,
                noise_decay=noise_decay,
            )
            all_rewards.append(ep_reward)
            if a_loss is not None:
                all_a_loss.append(a_loss)

            if c_loss is not None:
                all_c_loss.append(c_loss) 
                                            
                                        
            wandb.log({
                        "episode_reward": ep_reward/num_agents,
                        "actor_loss": a_loss if a_loss is not None else 0.0,
                        "critic_loss": c_loss if c_loss is not None else 0.0
                    }, step=episode)

            # ─── logging every 100 episodes ────────────────────────────────────
            if (episode+1) % 100 == 0:
                avg_reward = np.mean(all_rewards[-100:])/num_agents
                avg_a_loss = np.mean(all_a_loss[-100:])
                avg_c_loss = np.mean(all_c_loss[-100:])
                print(f"[Episode {episode:5d}]  AvgReward(100) = {avg_reward:8.2f}")
                wandb.log({
                    "avg_reward_100": avg_reward,
                    "avg_actor_loss_100": avg_a_loss,
                    "avg_critic_loss_100": avg_c_loss
                    },  step=episode)

                # checkpoint
                ckpt_dir = os.path.join("weights", task_name, "AC")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_filename = (
                    f"ac_agents{num_agents}_ep{episode}"
                    f"_lr{learning_rate}"
                    f"_bs{batch_size}"
                    f"_dis{discount_factor}"
                    f"_τ{tau}"
                    f"_hd{hidden_dim}.pt"
                )
                ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                torch.save(agent.actor.state_dict(), ckpt_path)
                print(f"Checkpoint saved → {ckpt_path}")
        
        print('Training Complete')
        # agent.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
            
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
    main() # type: ignore
    # close sim app
    simulation_app.close()
    wandb.finish()
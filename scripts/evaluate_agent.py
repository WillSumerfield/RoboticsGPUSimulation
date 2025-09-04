"""Evaluate the average performance of a single agent (SB3 or population_evolution) with given parameters, using a workflow similar to train.py."""

import argparse
import sys
import os
import numpy as np
import yaml
from datetime import datetime
from isaaclab.app import AppLauncher

import gymnasium as gym
import torch
from stable_baselines3 import PPO

parser = argparse.ArgumentParser(description="Evaluate an agent's average performance.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument('--agent_type', type=str, choices=['sb3', 'evolution'], required=True, help='Type of agent: sb3 or evolution')
parser.add_argument('--train_date', type=str, help='The date of the evolution run (for evolution type)')
parser.add_argument('--generation', type=str, help='The generation of the agent to evaluate (for evolution type)')
parser.add_argument('--agent_id', type=str, help='ID of the evolution agent to evaluate (for evolution type)')
parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes (default: num_envs)')
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse App
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
import IsaacEnv.tasks


def patch_usd_path_in_cfg(env_cfg, new_usd_name="Grasp3D.usd"):
    """Replace Grasp3D-temp.usd with Grasp3D.usd in the config."""
    if hasattr(env_cfg, 'robot_cfg') and hasattr(env_cfg.robot_cfg, 'spawn'):
        usd_path = env_cfg.robot_cfg.spawn.usd_path
        if "Grasp3D-temp.usd" in usd_path:
            env_cfg.robot_cfg.spawn.usd_path = usd_path.replace("Grasp3D-temp.usd", new_usd_name)
    return env_cfg


def evaluate_policy(agent, env, num_episodes):
    """Evaluate the agent for num_episodes and return average reward."""

    rewards = np.zeros((num_episodes, env.num_envs))
    for ep in range(num_episodes):
        obs = env.reset()
        done_indices = np.zeros(env.num_envs, dtype=np.bool_)

        while not done_indices.all():
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            rewards[ep, ~done_indices] += reward[~done_indices]
            done_indices |= done
        print(f"Episode {ep + 1}/{num_episodes} complete - {rewards[ep].mean():.3f} reward, {rewards[ep].std():.3f} std")

    return np.mean(rewards), np.std(rewards)


def evaluate_evolution_agent(agent_info, env_cfg, model_path, task_name, num_episodes):
    """Evaluate a population_evolution agent with given parameters."""
    from evolution_common import modify_params_usd
    modify_params_usd(task_name.replace("-", "_"), np.array(agent_info['params']))
    # Create environment
    env = gym.make(task_name, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = Sb3VecEnvWrapper(env)
    # Load model if available
    agent = PPO.load(model_path, env=env)
    avg_reward, std_reward = evaluate_policy(agent, env, num_episodes)
    env.close()
    return avg_reward, std_reward


@hydra_task_config(args.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):

    if args.num_envs is None:
        raise ValueError("num_envs must be specified in the config or command line arguments.")

    # Set up environment config
    env_cfg.scene.num_envs = getattr(args, 'num_envs', env_cfg.scene.num_envs)
    env_cfg.seed = 200

    if args.agent_type == 'sb3':
        log_dir = os.path.abspath(os.path.join("logs", "sb3", args.task))
        # Use the provided train date or default to latest
        train_date = args.train_date
        if not args.train_date:
            train_date = sorted(os.listdir(log_dir))[-1]
        model_path = f"logs/sb3/{args.task}/{train_date}/model"
        # Patch USD path for SB3 agent
        env_cfg = patch_usd_path_in_cfg(env_cfg, new_usd_name="Grasp3D.usd")
        # Post-process agent config
        agent_cfg = process_sb3_cfg(agent_cfg)
        # Create environment
        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = Sb3VecEnvWrapper(env)
        # Load model
        model = PPO.load(model_path, env=env)
        print(f"Running evaluation for {args.task} - {train_date}")
        avg_reward, std_reward = evaluate_policy(model, env, args.episodes)
        env.close()

    else:
        log_dir = os.path.abspath(os.path.join("logs", "sb3", "population_evolution", args.task.replace('-', '_').lower()))
        # Use the provided train date or default to latest
        train_date = args.train_date
        if not args.train_date:
            train_date = sorted(os.listdir(log_dir))[-1]
        # Find the results directory
        pop_dir = f"logs/sb3/population_evolution/{args.task.replace('-', '_').lower()}/{train_date}"
        model_path = pop_dir + f"/gen_{args.generation}/agent_{args.agent_id}/agent_{args.agent_id}_gen_{args.generation}"
        agent_info_path = pop_dir + f"/population_results.yaml"

        # Load agent info from YAML
        with open(agent_info_path, 'r') as f:
            agent_info = yaml.unsafe_load(f)
        
        # Find the specified agent
        agent_data = None
        for agent in agent_info['agents']:
            if str(agent['id']) == args.agent_id:
                agent_data = agent
                break
        print(f"Running evaluation for {args.task} - {train_date}: agent {agent_data['id']}, generation {args.generation}")
        avg_reward, std_reward = evaluate_evolution_agent(agent_data, env_cfg, model_path, args.task, args.episodes)

    print(f"Average reward over {args.episodes} episodes: {avg_reward:.3f} ± {std_reward:.3f}")
    simulation_app.close()

if __name__ == "__main__":
    main()

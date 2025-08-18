"""Agent Visualization Script.

This script allows you to visualize and optionally record a video of a specific agent
from your evolution training results. You can load any agent's parameters from any
generation and watch their performance in the environment.

Usage:
    python scripts/visualize_agent.py --log_dir logs/sb3/population_evolution/2025-07-30_14-30-15 --agent_id 3 --generation 5 --record
"""

import argparse
import sys
import os
import gc
import numpy as np
import yaml
import torch
import time
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize a trained agent from evolution results.")
parser.add_argument("--log_dir", type=str, required=True, help="Path to the evolution log directory.")
parser.add_argument("--agent_id", type=int, required=True, help="ID of the agent to visualize.")
parser.add_argument("--generation", type=int, required=True, help="Generation number to load the agent from.")
parser.add_argument("--task", type=str, default="Isaac-Isaacenv-Direct-v0", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run.")
parser.add_argument("--record", action="store_true", help="Record video of the agent's performance.")
parser.add_argument("--video_length", type=int, default=500, help="Length of recorded video in steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Override headless setting if user wants to record or visualize
if not args_cli.headless and not args_cli.record:
    args_cli.headless = False  # Show GUI for visualization

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, Gf

from stable_baselines3 import PPO

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_yaml
import isaacsim.core.utils.prims as prim_utils

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import IsaacEnv.tasks  # noqa: F401

from evolution_common import modify_params_usd


NUM_PARAMS = 3
PARAMETERIZED_PRIM_NAMES = [
    "/World/robot/right_lower_digit",
    "/World/robot/left_lower_digit",
    "/World/robot/back_lower_digit",
]


def load_agent_data(log_dir: str, agent_id: int):
    """Load agent parameters and configuration from evolution results."""
    
    # Load population results
    results_file = os.path.join(log_dir, "population_results.yaml")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = yaml.unsafe_load(f)
    
    # Find the specified agent
    agent_data = None
    for agent in results['agents']:
        if agent['id'] == agent_id:
            agent_data = agent
            break
    
    if agent_data is None:
        raise ValueError(f"Agent {agent_id} not found in results.")
    
    print(f"[INFO] Found Agent {agent_id}")
    print(f"  - Parameters: {agent_data['params']}")
    print(f"  - Best fitness: {agent_data['best_fitness']:.3f}")
    print(f"  - Generations trained: {len(agent_data['fitness_history'])}")
    
    return agent_data, results['config']


def load_agent_model(log_dir: str, agent_id: int, generation: int):
    """Load the trained model for the specified agent and generation."""
    
    model_dir = os.path.join(log_dir, f"gen_{generation}", f"agent_{agent_id}")
    model_path = os.path.join(model_dir, f"agent_{agent_id}_gen_{generation}")
    
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")
    
    print(f"[INFO] Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    return model


def create_environment(task_name: str, env_cfg, record=False, video_length=500, num_envs=1):
    """Create and configure the environment."""
    
    # Override config for single environment visualization
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, 'sim'):
        env_cfg.sim.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    render_mode = "rgb_array" if record else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

    if record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = f"videos/agent_visualization/{timestamp}"
        os.makedirs(video_dir, exist_ok=True)
        
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,  # Record from the start
            "video_length": video_length,
            "disable_logger": True,
        }
        
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env)
    
    # Don't wrap for stable baselines since we're just visualizing
    return env


def run_agent_episodes(model, env, num_episodes: int):
    """Run episodes with the loaded agent and optionally record video."""
    
    print(f"[INFO] Running {num_episodes} episodes...")
    
    obs = env.reset()
    episode_rewards = []
    episode_lengths = []
    for episode in range(num_episodes):
        print(f"[INFO] Running episode {episode + 1}/{num_episodes}")
        
        episode_length = 0
        episode_reward = 0.0
        done = False

        while not done:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            
            episode_length += 1
            episode_reward += reward.mean().item()
            
            # Add small delay for better visualization (if not headless)
            if not args_cli.headless:
                time.sleep(0.01)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {episode + 1}: Length = {episode_length}, Reward = {episode_reward:.3f}")
    
    avg_length = np.mean(episode_lengths)
    avg_reward = np.mean(episode_rewards)
    
    print(f"\n[RESULTS] Episode Summary:")
    print(f"  Average Length: {avg_length:.1f}")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Total Episodes: {num_episodes}")


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Main visualization function."""
    
    print(f"[INFO] Agent Visualization Tool")
    print(f"  Log Directory: {args_cli.log_dir}")
    print(f"  Agent ID: {args_cli.agent_id}")
    print(f"  Generation: {args_cli.generation}")

    task_folder = args_cli.task.replace("-", "_").lower()
    
    # Load agent data from evolution results
    agent_data, config = load_agent_data(args_cli.log_dir, args_cli.agent_id)
    
    # Load the trained model
    model = load_agent_model(args_cli.log_dir, args_cli.agent_id, args_cli.generation)
    
    # Set environment parameters
    modify_params_usd(task_folder, agent_data['params'])
    
    # Create environment
    print(f"[INFO] Creating environment...")
    env = create_environment(args_cli.task, env_cfg, record=args_cli.record, video_length=args_cli.video_length, num_envs=args_cli.num_envs)
    
    # Run agent episodes
    run_agent_episodes(
        model=model,
        env=env,
        num_episodes=args_cli.num_episodes
    )
    
    # Cleanup
    env.close()
    print(f"[INFO] Visualization complete!")


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    finally:
        simulation_app.close()

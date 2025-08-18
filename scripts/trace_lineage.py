"""Agent Lineage Tracer Script.

This script traces the family history of a specific agent and takes pictures
of the robot morphology at each mutation step in the lineage. It creates
a visual family tree showing how the robot's shape evolved over generations.

Usage:
    python scripts/trace_lineage.py --log_dir logs/sb3/population_evolution/2025-07-30_14-30-15 --agent_id 15
"""

import argparse
import sys
import os
import gc
import numpy as np
from sympy import capture
import yaml
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import asyncio

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Trace agent lineage and capture morphology images.")
parser.add_argument("--log_dir", type=str, required=True, help="Path to the evolution log directory.")
parser.add_argument("--agent_id", type=int, required=True, help="ID of the agent to trace lineage for.")
parser.add_argument("--task", type=str, default="Grasp-Object", help="Name of the task.")
parser.add_argument("--image_size", type=int, default=512, help="Size of captured images (width and height).")
parser.add_argument("--camera_distance", type=float, default=5.0, help="Distance of camera from the robot.")
parser.add_argument("--num_angles", type=int, default=1, help="Number of camera angles to capture (1-8).")
parser.add_argument("--output_dir", type=str, default="lineage_images", help="Output directory for images.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Force headless mode for image capture
#args_cli.headless = True

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
import omni.kit.commands
import carb

from stable_baselines3 import PPO

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.io import dump_yaml
import isaacsim.core.utils.prims as prim_utils

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import IsaacEnv.tasks  # noqa: F401

app = omni.kit.app.get_app()
extension_manager = app.get_extension_manager()
import omni.kit.viewport.utility as viewport_util


from evolution_common import modify_params_usd


NUM_PARAMS = 3
PARAMETERIZED_PRIM_NAMES = [
    "/World/robot/right_lower_digit",
    "/World/robot/left_lower_digit",
    "/World/robot/back_lower_digit",
]


def load_population_data(log_dir: str):
    """Load the complete population data from evolution results."""
    
    results_file = os.path.join(log_dir, "population_results.yaml")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = yaml.unsafe_load(f)
    
    return results


def trace_agent_lineage(population_data, target_agent_id):
    """Trace the complete lineage of an agent from root ancestor to target."""
    
    # Build agent lookup
    agents = {agent['id']: agent for agent in population_data['agents']}
    
    if target_agent_id not in agents:
        raise ValueError(f"Agent {target_agent_id} not found in population data.")
    
    target_agent = agents[target_agent_id]
    
    # Build the lineage chain
    lineage = []
    
    # Start with the target agent
    current_agent = target_agent
    lineage.append(current_agent)
    
    # Trace back through parents
    while current_agent.get('parent_id') is not None:
        parent_id = current_agent['parent_id']
        if parent_id not in agents:
            print(f"[WARNING] Parent {parent_id} not found in population data.")
            break
        
        current_agent = agents[parent_id]
        lineage.append(current_agent)
    
    # Reverse to get chronological order (root ancestor first)
    lineage.reverse()
    
    print(f"[INFO] Traced lineage for Agent {target_agent_id}:")
    print(f"  Total generations: {len(lineage)}")
    print(f"  Root ancestor: Agent {lineage[0]['id']}")
    print(f"  Lineage: {' → '.join([str(agent['id']) for agent in lineage])}")
    
    return lineage


def setup_camera_for_capture(image_size: int):
    """Setup camera for high-quality image capture using viewport."""
    
    print(f"[INFO] Setting up viewport camera with {image_size}x{image_size} resolution")
    
    # Get the active viewport
    viewport_api = viewport_util.get_active_viewport()
    
    # Set render resolution
    viewport_api.set_texture_resolution((image_size, image_size))
    
    return viewport_api


def capture_robot_images_viewport(viewport_api, output_dir: str, agent_info: dict, num_angles: int = 4):
    """Capture images using viewport API."""

    agent_id = agent_info['id']
    generation = agent_info['generation']
    
    # Create output directory for this agent
    agent_dir = os.path.abspath(os.path.join(output_dir, f"agent_{agent_id}_gen_{generation}"))
    os.makedirs(agent_dir, exist_ok=True)

    # Wait for the human to take a photo
    input("Press Enter once the image has been captured...")

    return 1


def create_lineage_summary(lineage, output_dir):
    """Create a summary image showing the complete lineage."""
    
    # Create a text summary
    summary_text = f"Agent Lineage Trace\n"
    summary_text += f"Target Agent: {lineage[-1]['id']}\n"
    summary_text += f"Total Generations: {len(lineage)}\n"
    summary_text += f"Root Ancestor: Agent {lineage[0]['id']}\n\n"
    
    summary_text += "Evolutionary Path:\n"
    for i, agent in enumerate(lineage):
        agent_id = agent['id']
        generation = agent['generation']
        params = agent['params']
        fitness = agent.get('best_fitness', 0)
        
        if i == 0:
            summary_text += f"Gen {generation}: Agent {agent_id} [ROOT]\n"
        else:
            parent_id = agent.get('parent_id', 'Unknown')
            summary_text += f"Gen {generation}: Agent {agent_id} ← Agent {parent_id}\n"

    # Save summary to file
    summary_path = os.path.join(output_dir, "lineage_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"[INFO] Lineage summary saved to: {summary_path}")
    
    return summary_path


def create_environment(task_name: str, env_cfg):
    """Create environment for morphology visualization."""
    
    # Override config for single environment
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg, 'sim'):
        env_cfg.sim.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    env = Sb3VecEnvWrapper(env)
    
    return env


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Main lineage tracing function."""
    
    print(f"[INFO] Agent Lineage Tracer")
    print(f"  Log Directory: {args_cli.log_dir}")
    print(f"  Target Agent: {args_cli.agent_id}")
    print(f"  Output Directory: {args_cli.output_dir}")

    task_folder = args_cli.task.replace("-", "_").lower()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args_cli.output_dir, task_folder, f"agent_{args_cli.agent_id}_lineage")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load population data
    print(f"[INFO] Loading population data...")
    population_data = load_population_data(args_cli.log_dir)
    
    # Trace agent lineage
    print(f"[INFO] Tracing lineage...")
    lineage = trace_agent_lineage(population_data, args_cli.agent_id)
    
    # Capture images for each agent in the lineage
    print(f"[INFO] Capturing morphology images...")
    image_count = 0
    
    for i, agent_info in enumerate(lineage):
        print(f"[INFO] Processing Agent {agent_info['id']} (step {i+1}/{len(lineage)})...")
        
        # Set morphology parameters
        modify_params_usd(task_folder, agent_info['params'])
        
        # Recreate the environment each run to apply new parameters
        print(f"[INFO] Creating environment...")
        env = create_environment(args_cli.task, env_cfg)
        obs = env.reset()

        # Setup camera
        print(f"[INFO] Setting up camera...")
        viewport_api = setup_camera_for_capture(args_cli.image_size)

        # Take a step to render the environment
        env.step(torch.zeros((1, env.action_space.shape[0]), device=env.unwrapped.device))

        # Wait for environment to stabilize
        time.sleep(0.5)
        
        # Capture images using viewport
        image_count += capture_robot_images_viewport(viewport_api, output_dir, agent_info, args_cli.num_angles)

        env.close()
        del env
    
    # Create lineage summary
    summary_path = create_lineage_summary(lineage, output_dir)
    
    print(f"\n[SUCCESS] Lineage tracing complete!")
    print(f"  Images captured: {image_count}")
    print(f"  Output directory: {output_dir}")
    print(f"  Summary file: {summary_path}")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    finally:
        simulation_app.close()

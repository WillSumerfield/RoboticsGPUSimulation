"""Lineage training script.

Takes a lineage from population evolution results and trains each mutation step
for the full amount of timesteps it would have received during evolution.
This allows us to see the true potential of each step in the lineage.
"""

import argparse
import sys
import os
import gc
import glob
import numpy as np
from datetime import datetime
import yaml
import torch

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Train each step of a lineage for full evolution timesteps.")
parser.add_argument("--results_file", type=str, required=True, help="Path to population_results.yaml file.")
parser.add_argument("--agent_id", type=int, required=True, help="ID of the agent whose lineage to train.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Grasp-Object", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
import omni.timeline
import omni.usd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml, load_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import IsaacEnv.tasks  # noqa: F401

from scripts.evolution_common import modify_params_usd, Agent


def recreate_environment(env_cfg, task_name):
    """Recreate the environment to apply new parameters."""
    # Clean shutdown
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.commit()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create new environment
    env = gym.make(task_name, cfg=env_cfg)
    
    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Wrap for stable baselines
    env = Sb3VecEnvWrapper(env)
    
    return env


def find_agent_lineage(results_data, target_agent_id):
    """Find the complete lineage of an agent from the results data."""
    agents = results_data['agents']
    agent_dict = {agent['id']: agent for agent in agents}
    
    if target_agent_id not in agent_dict:
        raise ValueError(f"Agent {target_agent_id} not found in results")
    
    target_agent = agent_dict[target_agent_id]
    
    # Build lineage by following family tree
    lineage = []
    
    # Start with the target agent and work backwards through family tree
    current_agent = target_agent
    
    # Add the target agent first
    lineage.append({
        'agent_id': current_agent['id'],
        'generation': current_agent['generation'],
        'parent_id': current_agent['parent_id'],
        'params': current_agent['params'],
        'original_fitness': current_agent['best_fitness']
    })
    
    # Follow the family tree backwards
    family_tree = current_agent['family_tree']
    for gen, parent_id in reversed(family_tree):
        if parent_id in agent_dict:
            parent_agent = agent_dict[parent_id]
            lineage.append({
                'agent_id': parent_agent['id'],
                'generation': parent_agent['generation'],
                'parent_id': parent_agent['parent_id'],
                'params': parent_agent['params'],
                'original_fitness': parent_agent['best_fitness']
            })
    
    # Reverse to get chronological order (oldest ancestor first)
    lineage.reverse()
    
    return lineage


def find_agent_checkpoint(base_results_dir, agent_id, generation):
    """Find the checkpoint file for a specific agent at a specific generation."""
    # The checkpoint should be in: base_results_dir/gen_{generation}/agent_{agent_id}/
    checkpoint_dir = os.path.join(base_results_dir, f"gen_{generation}", f"agent_{agent_id}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Look for checkpoint files (try different naming patterns)
    possible_names = [
        f"agent_{agent_id}_gen_{generation}_*_steps.zip",
        f"agent_{agent_id}_gen_{generation}.zip",
        "*.zip"
    ]
    
    for pattern in possible_names:
        checkpoint_path = os.path.join(checkpoint_dir, pattern)
        matches = glob.glob(checkpoint_path)
        if matches:
            # Use the most recent checkpoint if multiple exist
            latest_checkpoint = max(matches, key=os.path.getctime)
            print(f"Found checkpoint: {latest_checkpoint}")
            return latest_checkpoint
    
    print(f"Warning: No checkpoint found for agent {agent_id} generation {generation}")
    return None


def calculate_additional_timesteps(agent_generation, total_generations, timesteps_per_agent):
    """Calculate how many additional timesteps this agent would have received if it survived."""
    # Agent was trained once at its generation, so it would get training in remaining generations
    remaining_generations = total_generations - agent_generation
    additional_timesteps = remaining_generations * timesteps_per_agent
    return max(0, additional_timesteps)  # Ensure non-negative


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Main lineage training loop."""
    
    # Load results
    results_data = load_yaml(args_cli.results_file)
    config = results_data['config']
    
    # Get evolution parameters
    total_generations = config['generations']
    timesteps_per_agent = config['timesteps_per_agent']
    
    print(f"Loading lineage for agent {args_cli.agent_id}")
    print(f"Original evolution: {total_generations} generations × {timesteps_per_agent} timesteps")
    
    # Find the lineage
    lineage = find_agent_lineage(results_data, args_cli.agent_id)
    
    print(f"Found lineage with {len(lineage)} steps:")
    for i, step in enumerate(lineage):
        additional_timesteps = calculate_additional_timesteps(step['generation'], total_generations, timesteps_per_agent)
        print(f"  Step {i}: Agent {step['agent_id']} (gen {step['generation']}, fitness: {step['original_fitness']:.3f}) - {additional_timesteps} additional timesteps")
    
    # Setup
    rng = np.random.default_rng(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    task_folder = args_cli.task.replace("-", "_").lower()
    
    # Setup logging
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_results_dir = os.path.dirname(args_cli.results_file)
    log_dir = os.path.join(base_results_dir, "lineage_training", f"agent_{args_cli.agent_id}", run_info)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Results will be saved to: {log_dir}")
    
    # Process agent config
    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    agent_cfg.pop("n_timesteps", None)
    
    # Train each step in the lineage
    lineage_results = []
    
    for step_idx, lineage_step in enumerate(lineage):
        additional_timesteps = calculate_additional_timesteps(lineage_step['generation'], total_generations, timesteps_per_agent)
        
        print(f"\n{'='*60}")
        print(f"Training lineage step {step_idx + 1}/{len(lineage)}")
        print(f"Agent {lineage_step['agent_id']} (Generation {lineage_step['generation']})")
        print(f"Original fitness: {lineage_step['original_fitness']:.3f}")
        print(f"Additional timesteps to train: {additional_timesteps}")
        print(f"{'='*60}")
        
        # Convert params back to numpy array
        params = np.array(lineage_step['params'])
        
        # Set parameters and recreate environment
        print("Setting up environment...")
        modify_params_usd(task_folder, params)
        env = recreate_environment(env_cfg, args_cli.task)
        print("Environment ready")
        
        # Create agent
        agent = Agent(
            lineage_step['agent_id'], 
            params, 
            policy_arch, 
            agent_cfg, 
            lineage_step['generation'],
            lineage_step['parent_id']
        )
        
        # Create model
        agent.update_model(env)
        
        # Try to load checkpoint from original evolution
        checkpoint_path = find_agent_checkpoint(base_results_dir, lineage_step['agent_id'], lineage_step['generation'])
        
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            try:
                agent.model = PPO.load(checkpoint_path, env=env)
                print("Successfully loaded checkpoint")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch")
        else:
            print("No checkpoint found, starting training from scratch")
            
        # Skip training if no additional timesteps needed
        if additional_timesteps <= 0:
            print("No additional training needed for this agent")
            final_fitness = lineage_step['original_fitness']
        else:
            # Setup logging for this lineage step
            step_log_dir = os.path.join(log_dir, f"step_{step_idx:02d}_agent_{lineage_step['agent_id']}")
            os.makedirs(step_log_dir, exist_ok=True)
            
            logger = configure(step_log_dir, [])
            agent.model.set_logger(logger)
            
            # Train for additional timesteps
            checkpoint_callback = CheckpointCallback(
                save_freq=max(1000, additional_timesteps // 10) if additional_timesteps > 0 else 1000,
                save_path=step_log_dir,
                name_prefix=f"lineage_step_{step_idx}",
                verbose=1
            )
            
            print(f"Training for {additional_timesteps} additional timesteps...")
            if additional_timesteps > 0:
                agent.train(additional_timesteps, callback=checkpoint_callback)
            print("Training complete")
            
            # Evaluate final fitness
            print("Evaluating final fitness...")
            final_fitness = agent.evaluate_fitness(env, generation=0, num_episodes=5)  # More episodes for better estimate
            print(f"Final fitness: {final_fitness:.3f}")
            
            # Save model
            model_path = os.path.join(step_log_dir, f"lineage_step_{step_idx}_final")
            agent.model.save(model_path)
        
        # Store results
        step_result = {
            'step_index': step_idx,
            'agent_id': lineage_step['agent_id'],
            'generation': lineage_step['generation'],
            'parent_id': lineage_step['parent_id'],
            'params': lineage_step['params'],
            'original_fitness': lineage_step['original_fitness'],
            'trained_fitness': float(final_fitness),
            'improvement': float(final_fitness - lineage_step['original_fitness']),
            'additional_timesteps': additional_timesteps,
            'checkpoint_found': checkpoint_path is not None,
            'checkpoint_path': checkpoint_path,
            'model_path': model_path if additional_timesteps > 0 else None
        }
        
        lineage_results.append(step_result)
        
        print(f"Fitness improvement: {step_result['improvement']:+.3f}")
        if checkpoint_path:
            print(f"Started from checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Clean up environment
        env.close()
        del env
        del agent
    
    # Calculate summary statistics
    total_additional_timesteps = sum(step['additional_timesteps'] for step in lineage_results)
    checkpoints_found = sum(1 for step in lineage_results if step['checkpoint_found'])
    
    # Save lineage training results
    results = {
        'target_agent_id': args_cli.agent_id,
        'lineage_steps': lineage_results,
        'config': {
            'original_config': config,
            'task': args_cli.task,
            'num_envs': args_cli.num_envs,
            'seed': args_cli.seed,
            'base_results_dir': base_results_dir
        },
        'summary': {
            'lineage_length': len(lineage_results),
            'best_original_fitness': max(step['original_fitness'] for step in lineage_results),
            'best_trained_fitness': max(step['trained_fitness'] for step in lineage_results),
            'max_improvement': max(step['improvement'] for step in lineage_results),
            'total_additional_timesteps': total_additional_timesteps,
            'checkpoints_found': checkpoints_found,
            'checkpoints_total': len(lineage_results)
        }
    }
    
    results_path = os.path.join(log_dir, "lineage_training_results.yaml")
    dump_yaml(results_path, results)
    
    print(f"\n{'='*60}")
    print("LINEAGE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Lineage length: {results['summary']['lineage_length']} steps")
    print(f"Checkpoints found: {checkpoints_found}/{len(lineage_results)}")
    print(f"Total additional training: {total_additional_timesteps:,} timesteps")
    print(f"Best original fitness: {results['summary']['best_original_fitness']:.3f}")
    print(f"Best trained fitness: {results['summary']['best_trained_fitness']:.3f}")
    print(f"Maximum improvement: {results['summary']['max_improvement']:+.3f}")
    
    # Print step-by-step results
    print(f"\nStep-by-step results:")
    for step in lineage_results:
        checkpoint_status = "✓" if step['checkpoint_found'] else "✗"
        print(f"  Step {step['step_index']}: Agent {step['agent_id']} [{checkpoint_status}] - "
              f"Original: {step['original_fitness']:.3f}, "
              f"Trained: {step['trained_fitness']:.3f}, "
              f"Improvement: {step['improvement']:+.3f}, "
              f"Additional timesteps: {step['additional_timesteps']:,}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

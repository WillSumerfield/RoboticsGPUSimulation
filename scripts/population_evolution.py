"""Population-based evolution training script.

Trains a population of agents, each with different parameterized morphologies.
Each generation, agents are fine-tuned for a short period before
moving to the next agent with a different morphology configuration.
"""

import argparse
import sys
import os
import gc
import numpy as np
from datetime import datetime
import subprocess
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Train a population of RL agents with different morphology parameters.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Isaacenv-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--population_size", type=int, default=8, help="Number of agents in the population.")
parser.add_argument("--generations", type=int, default=10, help="Number of generations to evolve.")
parser.add_argument("--timesteps_per_agent", type=int, default=5000, help="Training timesteps per agent per generation.")
parser.add_argument("--min_scale", type=float, default=0.3, help="Minimum parameter scale factor.")
parser.add_argument("--max_scale", type=float, default=2.0, help="Maximum parameter scale factor.")

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
from pxr import Usd, UsdGeom, Gf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
import isaacsim.core.utils.prims as prim_utils

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import IsaacEnv.tasks  # noqa: F401


NUM_PARAMS = 3
PARAMETERIZED_PRIM_NAMES = [
    "/World/robot/right_lower_digit",
    "/World/robot/left_lower_digit",
    "/World/robot/back_lower_digit",
]
PARAMETERIZED_JOINT_NAMES= [
    "/World/robot/left_lower_digit/left_lower_fixed",
    "/World/robot/right_lower_digit/right_lower_fixed",
    "/World/robot/back_lower_digit/back_lower_fixed",
]
PARAM_OFFSET = 1/(2*np.sqrt(2))


class EvolutionLogger:
    """Custom logger for evolution progress that writes to both file and separate terminal."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logging
        self.log_file = os.path.join(log_dir, "evolution_progress.log")
        
        # Create empty log file first
        with open(self.log_file, 'w') as f:
            f.write("")
        
        self.file_logger = logging.getLogger("evolution")
        self.file_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.file_logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)
        
        # Open separate terminal for progress display
        self.terminal_process = None
        self._open_progress_terminal()
    
    def _open_progress_terminal(self):
        """Open a new terminal window to display progress."""
        try:
            # For Windows PowerShell
            if sys.platform == "win32":
                # Use Get-Content without -Tail to show all content from beginning
                cmd = [
                    "powershell", "-Command", 
                    f"Get-Content '{self.log_file}' -Wait"
                ]
                self.terminal_process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            # For Linux/Mac
            else:
                # Try different terminal emulators
                terminals = ["gnome-terminal", "xterm", "konsole"]
                for terminal in terminals:
                    try:
                        self.terminal_process = subprocess.Popen([
                            terminal, "--", "tail", "-f", self.log_file
                        ])
                        break
                    except FileNotFoundError:
                        continue
        except Exception as e:
            print(f"[WARNING] Could not open progress terminal: {e}")
            self.terminal_process = None
    
    def log(self, message: str):
        """Log message to both file and console."""
        # Log to file (which feeds the separate terminal)
        self.file_logger.info(message)
        
        # Force flush to ensure immediate write
        for handler in self.file_logger.handlers:
            handler.flush()
        
        # Also log to main console for backup
        print(f"[EVOLUTION] {message}")
    
    def close(self):
        """Close the logger and terminal."""
        if self.terminal_process:
            try:
                self.terminal_process.terminate()
            except:
                pass


class Agent:
    """Represents an agent with its policy and parameters."""
    
    def __init__(self, agent_id: int, params: tuple, policy_arch: str, agent_cfg: dict):
        self.id = agent_id
        self.params = params
        self.policy_arch = policy_arch
        self.agent_cfg = agent_cfg.copy()
        self.model = None
        self.fitness_history = []
    
    def update_model(self, env):
        if self.model is None:
            self.model = PPO(self.policy_arch, env, **self.agent_cfg)
        else:
            new_model = PPO(self.policy_arch, env, **self.agent_cfg)
            new_model.set_parameters(self.model.get_parameters())
            del self.model
            self.model = new_model

    def train(self, timesteps: int, callback=None):
        """Train the agent for specified timesteps."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
    
    def evaluate_fitness(self, env, num_episodes: int = 1):
        """Evaluate agent fitness by running episodes and measuring performance."""
        if self.model is None:
            return 0.0
        
        total_reward = 0.0
        for _ in range(num_episodes):
            obs = env.reset()
            done_count = 0
            episode_reward = 0.0
            
            while done_count < env.num_envs * num_episodes:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward.mean() if hasattr(reward, 'mean') else reward
                done_count += done.sum().item()
            
            total_reward += episode_reward
        
        fitness = total_reward / num_episodes
        self.fitness_history.append(fitness)
        return fitness


def modify_params_usd(parameters: tuple):
    """Modify the USD file with new parameters."""
    stage = Usd.Stage.Open("source/IsaacEnv/IsaacEnv/tasks/direct/isaacenv/Grasp3D-v3.usd")
    prims = [stage.GetPrimAtPath(prim_name) for prim_name in PARAMETERIZED_PRIM_NAMES]
    
    # Scale each prim to the new scale factor
    for param, prim in zip(parameters, prims):
        prim_xform = UsdGeom.Xformable(prim)

        # Find or create scale operation
        scale_op = None
        for op in prim_xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xform.AddScaleOp()

        # Set the length
        scale = scale_op.Get()
        scale[2] = param # Z is the length of the digits
        scale_op.Set(scale)
    
    stage.GetRootLayer().Export("source/IsaacEnv/IsaacEnv/tasks/direct/isaacenv/Grasp3D-temp.usd")

    # Reload the USD file using Omniverse context
    context = omni.usd.get_context()
    context.new_stage()
    context.get_stage().Reload()


def create_evolution_plots(population, log_dir, evolution_logger):
    """Create and save evolution performance plots."""
    evolution_logger.log("Creating performance visualization plots...")
    
    # Set seaborn style and cool color palette
    sns.set_style("whitegrid")
    sns.set_palette("cool", len(population))
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Prepare data for plotting
    plot_data = []
    for agent in population:
        for gen, fitness in enumerate(agent.fitness_history):
            plot_data.append({
                'Agent': f'Agent {agent.id}',
                'Generation': gen + 1,
                'Fitness': fitness,
                'Parameters': f"({agent.params[0]:.2f}, {agent.params[1]:.2f}, {agent.params[2]:.2f})"
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Population Evolution Analysis', fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Plot 1: Fitness evolution over generations for each agent
    sns.lineplot(data=df, x='Generation', y='Fitness', hue='Agent', 
                marker='o', linewidth=2.5, markersize=6, ax=ax1)
    ax1.set_title('Fitness Evolution by Agent', fontsize=14, fontweight='bold', color='#34495e')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
               fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best fitness per generation
    best_fitness_per_gen = df.groupby('Generation')['Fitness'].max().reset_index()
    sns.lineplot(data=best_fitness_per_gen, x='Generation', y='Fitness', 
                marker='s', linewidth=3, markersize=8, color='#e74c3c', ax=ax2)
    ax2.set_title('Best Fitness per Generation', fontsize=14, fontweight='bold', color='#34495e')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Best Fitness', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final fitness distribution
    final_fitness = [agent.fitness_history[-1] if agent.fitness_history else 0 for agent in population]
    agent_names = [f'Agent {agent.id}' for agent in population]
    
    bars = ax3.bar(agent_names, final_fitness, color=sns.color_palette("cool", len(population)))
    ax3.set_title('Final Fitness Distribution', fontsize=14, fontweight='bold', color='#34495e')
    ax3.set_xlabel('Agent', fontsize=12)
    ax3.set_ylabel('Final Fitness', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, fitness in zip(bars, final_fitness):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{fitness:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Average fitness per generation with confidence interval
    avg_fitness_data = []
    for gen in range(1, len(population[0].fitness_history) + 1):
        gen_fitness = [agent.fitness_history[gen-1] for agent in population if len(agent.fitness_history) >= gen]
        if gen_fitness:
            avg_fitness_data.append({
                'Generation': gen,
                'Mean_Fitness': np.mean(gen_fitness),
                'Std_Fitness': np.std(gen_fitness)
            })
    
    avg_df = pd.DataFrame(avg_fitness_data)
    ax4.plot(avg_df['Generation'], avg_df['Mean_Fitness'], 
            color='#3498db', linewidth=3, marker='o', markersize=6, label='Mean')
    ax4.fill_between(avg_df['Generation'], 
                    avg_df['Mean_Fitness'] - avg_df['Std_Fitness'],
                    avg_df['Mean_Fitness'] + avg_df['Std_Fitness'],
                    alpha=0.3, color='#3498db')
    ax4.set_title('Population Average Fitness ± Std', fontsize=14, fontweight='bold', color='#34495e')
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Average Fitness', fontsize=12)
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "evolution_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create parameter correlation plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Prepare parameter correlation data
    param_data = []
    for agent in population:
        final_fitness = agent.fitness_history[-1] if agent.fitness_history else 0
        param_data.append({
            'Agent_ID': agent.id,
            'Param_1': agent.params[0],
            'Param_2': agent.params[1], 
            'Param_3': agent.params[2],
            'Final_Fitness': final_fitness
        })
    
    param_df = pd.DataFrame(param_data)
    
    # Create correlation heatmap
    corr_matrix = param_df[['Param_1', 'Param_2', 'Param_3', 'Final_Fitness']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Parameter-Fitness Correlation Matrix', 
                fontsize=14, fontweight='bold', color='#34495e')
    
    plt.tight_layout()
    corr_path = os.path.join(log_dir, "parameter_correlation.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    evolution_logger.log(f"Plots saved:")
    evolution_logger.log(f"  - Evolution analysis: {plot_path}")
    evolution_logger.log(f"  - Parameter correlation: {corr_path}")
    
    return plot_path, corr_path


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


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Main evolution training loop."""
    
    # Setup
    rng = np.random.default_rng(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # Setup logging
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "sb3", "population_evolution", run_info)
    
    # Initialize evolution logger
    evolution_logger = EvolutionLogger(log_dir)
    
    evolution_logger.log("Population Evolution Training Started")
    evolution_logger.log(f"Population size: {args_cli.population_size}")
    evolution_logger.log(f"Generations: {args_cli.generations}")
    evolution_logger.log(f"Timesteps per agent: {args_cli.timesteps_per_agent}")
    evolution_logger.log(f"Parameter range: {args_cli.min_scale} - {args_cli.max_scale}")
    evolution_logger.log(f"Logging to: {log_dir}")
    
    # Process agent config
    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    agent_cfg.pop("n_timesteps", None)
    
    # Create population with random parameters
    population = []
    evolution_logger.log("Creating population:")
    for i in range(args_cli.population_size):
        # Generate random parameters
        params = [rng.uniform(args_cli.min_scale, args_cli.max_scale) for _ in range(NUM_PARAMS)]
        
        agent = Agent(i, params, policy_arch, agent_cfg)
        population.append(agent)
        evolution_logger.log(f"Agent {i}: Params ({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f})")

    # Evolution loop
    for generation in range(args_cli.generations):
        evolution_logger.log(f"{'='*50}")
        evolution_logger.log(f"GENERATION {generation + 1}/{args_cli.generations}")
        evolution_logger.log(f"{'='*50}")
        
        generation_fitness = []
        
        for agent in population:
            evolution_logger.log(f"Training Agent {agent.id} (params: {agent.params})")

            # Set parameters and recreate environment
            evolution_logger.log("  Setting up environment...")
            modify_params_usd(agent.params)
            env = recreate_environment(env_cfg, args_cli.task)
            evolution_logger.log("  Environment ready")
            
            # Create/update agent model
            agent.update_model(env)
            
            # Setup logging for this agent
            agent_log_dir = os.path.join(log_dir, f"gen_{generation+1}", f"agent_{agent.id}")
            
            logger = configure(agent_log_dir, [])  # No console output
            agent.model.set_logger(logger)
            
            # Train agent
            checkpoint_callback = CheckpointCallback(
                save_freq=max(1000, args_cli.timesteps_per_agent // 5),
                save_path=agent_log_dir,
                name_prefix=f"agent_{agent.id}_gen_{generation+1}",
                verbose=0
            )
            
            evolution_logger.log(f"  Training for {args_cli.timesteps_per_agent} timesteps...")
            agent.train(args_cli.timesteps_per_agent, callback=checkpoint_callback)
            evolution_logger.log("  Training complete")
            
            # Evaluate fitness
            evolution_logger.log("  Evaluating fitness...")
            fitness = agent.evaluate_fitness(env)
            generation_fitness.append(fitness)
            
            evolution_logger.log(f"  Fitness: {fitness:.3f}")
            
            # Save model
            model_path = os.path.join(agent_log_dir, f"agent_{agent.id}_gen_{generation+1}")
            agent.model.save(model_path)
            
            # Clean up environment
            env.close()
            del env
        
        # Report generation statistics
        avg_fitness = np.mean(generation_fitness)
        best_fitness = np.max(generation_fitness)
        best_agent_id = np.argmax(generation_fitness)
        
        evolution_logger.log(f"Generation {generation + 1} Results:")
        evolution_logger.log(f"  Average fitness: {avg_fitness:.3f}")
        evolution_logger.log(f"  Best fitness: {best_fitness:.3f} (Agent {best_agent_id})")
        evolution_logger.log(f"  Best agent: {best_agent_id}")

    # Final report
    evolution_logger.log(f"{'='*50}")
    evolution_logger.log("EVOLUTION COMPLETE")
    evolution_logger.log(f"{'='*50}")
    
    # Find overall best agent
    all_fitness = [max(agent.fitness_history) for agent in population]
    best_overall_id = np.argmax(all_fitness)
    best_agent = population[best_overall_id]
    
    evolution_logger.log(f"Best overall agent: {best_agent.id}")
    evolution_logger.log(f"Best fitness: {max(best_agent.fitness_history):.3f}")
    evolution_logger.log(f"Best params: {best_agent.params}")

    # Save population data
    population_data = {
        "agents": [
            {
                "id": int(agent.id),
                "params": [float(p) for p in agent.params],
                "fitness_history": [float(f) for f in agent.fitness_history],
                "best_fitness": float(max(agent.fitness_history)) if agent.fitness_history else 0.0
            }
            for agent in population
        ],
        "config": {
            "population_size": int(args_cli.population_size),
            "generations": int(args_cli.generations),
            "timesteps_per_agent": int(args_cli.timesteps_per_agent),
            "scale_range": [float(args_cli.min_scale), float(args_cli.max_scale)]
        }
    }
    
    dump_yaml(os.path.join(log_dir, "population_results.yaml"), population_data)
    evolution_logger.log(f"Results saved to {log_dir}")
    
    # Create and save evolution plots
    plot_paths = create_evolution_plots(population, log_dir, evolution_logger)
    
    # Close evolution logger
    evolution_logger.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

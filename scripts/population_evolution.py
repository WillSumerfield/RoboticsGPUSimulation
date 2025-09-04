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
np.set_printoptions(precision=2, suppress=True)

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Train a population of RL agents with different morphology parameters.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Grasp-Object", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--population_size", type=int, default=8, help="Number of agents in the population.")
parser.add_argument("--generations", type=int, default=10, help="Number of generations to evolve.")
parser.add_argument("--timesteps_per_agent", type=int, default=2.6e5, help="Training timesteps per agent per generation.")
parser.add_argument("--mutation_rate", type=float, default=0.5, help="Fraction of population to mutate each generation.")
parser.add_argument("--mutation_strength", type=float, default=0.1, help="Standard deviation of gaussian mutation noise.")
parser.add_argument("--selection_randomness", type=float, default=0.2, help="Amount of randomness in selection (0=pure elitism, 1=random).")
parser.add_argument("--disable_param_randomization", action="store_true", help="Disable initial randomization of parameters.")

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

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import IsaacEnv.tasks  # noqa: F401

from evolution_common import modify_params_usd, Agent


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
        # Use Get-Content without -Tail to show all content from beginning
        cmd = [
            "powershell", "-Command", 
            f"Get-Content '{self.log_file}' -Wait"
        ]
        self.terminal_process = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    
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


def create_evolution_plots(population, log_dir, evolution_logger):
    """Create and save evolution performance plots."""
    evolution_logger.log("Creating performance visualization plots...")
    
    # Set seaborn style and cool color palette
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Build generation-by-generation data correctly
    # Track which agents were active in each generation
    generation_stats = {}
    max_generation = 0
    
    for agent in population:
        if agent.fitness_history:
            for gen_idx, fitness in agent.fitness_history.items():
                max_generation = max(max_generation, gen_idx)
                
                if gen_idx not in generation_stats:
                    generation_stats[gen_idx] = []
                
                generation_stats[gen_idx].append({
                    'agent_id': agent.id,
                    'fitness': fitness,
                    'params': agent.params,
                    'parent_id': agent.parent_id,
                    'is_original': len(agent.family_tree) == 0
                })
    
    if not generation_stats:
        evolution_logger.log("No generation data to plot")
        return None, None
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Population Evolution Analysis', fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Plot 1: Fitness evolution over generations (FIXED)
    generations = sorted(generation_stats.keys())
    best_fitness = []
    avg_fitness = []
    worst_fitness = []
    
    for gen in generations:
        fitnesses = [agent['fitness'] for agent in generation_stats[gen]]
        best_fitness.append(max(fitnesses))
        avg_fitness.append(np.mean(fitnesses))
        worst_fitness.append(min(fitnesses))
    
    ax1.plot(generations, best_fitness, 'g-', linewidth=3, marker='o', label='Best', markersize=8)
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, marker='s', label='Average', markersize=6)
    ax1.plot(generations, worst_fitness, 'r-', linewidth=2, marker='^', label='Worst', markersize=6)
    ax1.fill_between(generations, worst_fitness, best_fitness, alpha=0.2, color='lightblue')
    ax1.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold', color='#34495e')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(generations)
    
    # Plot 2: Parameter diversity over generations
    param_diversity = []
    for gen in generations:
        params_matrix = np.array([agent['params'] for agent in generation_stats[gen]])
        if len(params_matrix) > 1:
            # Calculate diversity as average standard deviation across parameters
            diversity = np.mean(np.std(params_matrix, axis=0))
            param_diversity.append(diversity)
        else:
            param_diversity.append(0)
    
    ax2.plot(generations, param_diversity, 'purple', linewidth=3, marker='o', markersize=8)
    ax2.set_title('Parameter Diversity Over Generations', fontsize=14, fontweight='bold', color='#34495e')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Parameter Diversity (Std)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(generations)
    
    # Plot 3: Final generation fitness distribution (FIXED)
    final_gen = max(generations)
    final_agents = generation_stats[final_gen]
    
    # Sort agents by fitness for better visualization
    final_agents_sorted = sorted(final_agents, key=lambda x: x['fitness'], reverse=True)
    agent_labels = [f"Agent {agent['agent_id']}" for agent in final_agents_sorted]
    fitnesses = [agent['fitness'] for agent in final_agents_sorted]
    
    # Use distinct colors instead of a single color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(final_agents_sorted)))
    bars = ax3.bar(range(len(final_agents_sorted)), fitnesses, color=colors, edgecolor='black', linewidth=1)
    ax3.set_title(f'Final Generation (Gen {final_gen}) Fitness Distribution', fontsize=14, fontweight='bold', color='#34495e')
    ax3.set_xlabel('Agent (Ranked by Fitness)', fontsize=12)
    ax3.set_ylabel('Fitness', fontsize=12)
    ax3.set_xticks(range(len(final_agents_sorted)))
    ax3.set_xticklabels([f"A{agent['agent_id']}" for agent in final_agents_sorted], rotation=45)
    
    # Add fitness values on bars
    for i, (bar, agent) in enumerate(zip(bars, final_agents_sorted)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{agent["fitness"]:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Best Agent Parameters Evolution
    ax4.set_title('Best Agent Parameters Evolution', fontsize=14, fontweight='bold', color='#34495e')
    
    # Get parameter ranges for normalization
    param_ranges = [
        (Agent.MIN_SCALE_LENGTH, Agent.MAX_SCALE_LENGTH),  # Length
        (Agent.MIN_SCALE_WIDTH, Agent.MAX_SCALE_WIDTH),    # Width
    ]
    param_names = ['Length', 'Width']
    
    # Find the best agent from each generation and get their parameters
    elite_params_data = []
    for gen in generations:
        best_agent = max(generation_stats[gen], key=lambda x: x['fitness'])
        elite_params_data.append({
            'generation': gen,
            'params': best_agent['params'],
            'fitness': best_agent['fitness']
        })
    
    # Set up bar positions
    bar_width = 0.25
    generation_positions = np.arange(len(generations))
    
    # Colors for each parameter type
    param_colors = ['#ff6b6b', '#45b7d1']  # Red, Teal, Blue
    
    # Create normalized parameter data and plot bars
    for param_idx, (param_name, (min_val, max_val), color) in enumerate(zip(param_names, param_ranges, param_colors)):
        # Get parameter values for all generations (all digits combined)
        param_values = []
        raw_values = []
        
        for elite_data in elite_params_data:
            params = elite_data['params']
            # Get parameter values for all digits for this parameter type
            digit_values = params[:, param_idx]  # All digits for this parameter type
            
            # Normalize to 0-1 range
            normalized = (digit_values - min_val) / (max_val - min_val)
            
            # Average across digits for this generation
            avg_normalized = np.mean(normalized)
            avg_raw = np.mean(digit_values)
            
            param_values.append(avg_normalized)
            raw_values.append(avg_raw)
        
        # Plot bars for this parameter type
        bar_positions = generation_positions + (param_idx - 1) * bar_width
        bars = ax4.bar(bar_positions, param_values, bar_width, 
                      label=param_name, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add raw values as text on bars
        for i, (bar, raw_val) in enumerate(zip(bars, raw_values)):
            height = bar.get_height()
            if param_name == 'Rotation':
                # Convert radians to degrees for display
                display_val = np.degrees(raw_val)
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{display_val:.0f}°', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold', rotation=90)
            else:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{raw_val:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', rotation=90)
    
    # Customize the plot
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Normalized Parameter Value', fontsize=12)
    ax4.set_xticks(generation_positions)
    ax4.set_xticklabels([f'Gen {gen}' for gen in generations])
    ax4.set_ylim(0, 1.2)  # Leave space for text labels
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add explanation text
    ax4.text(0.02, 0.98, 'Bars show normalized values (0-1), text shows actual values', 
            fontsize=10, style='italic', color='gray', transform=ax4.transAxes, 
            verticalalignment='top')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "evolution_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    evolution_logger.log(f"Plots saved:")
    evolution_logger.log(f"  - Evolution analysis: {plot_path}")
    
    return plot_path, None


def select_survivors(population, num_survivors: int, selection_randomness: float, rng):
    """Select survivors for the next generation using tournament selection with randomness."""
    
    # Get fitness scores for all agents
    fitness_scores = []
    for agent in population:
        # Get fitness from the most recent generation
        last_generation = max(agent.fitness_history.keys())
        fitness_scores.append((agent, agent.fitness_history[last_generation]))
    
    # Sort by fitness (descending)
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    
    survivors = []
    
    # Pure elitism: take top performers
    elite_count = int(num_survivors * (1 - selection_randomness))
    survivors.extend([agent for agent, _ in fitness_scores[:elite_count]])
    
    # Random selection from remaining slots
    remaining_count = num_survivors - elite_count
    if remaining_count > 0:
        # Tournament selection for remaining slots
        remaining_pool = fitness_scores[elite_count:]
        for _ in range(remaining_count):
            if remaining_pool:
                # Select best from random tournament
                tournament_size = min(3, len(remaining_pool))
                tournament = rng.choice(remaining_pool, tournament_size, replace=False)
                winner = max(tournament, key=lambda x: x[1])
                survivors.append(winner[0])
                # Remove winner from pool to avoid duplicates
                remaining_pool = [x for x in remaining_pool if x[0].id != winner[0].id]
    
    return survivors


def mutate_population(survivors, generation: int, mutation_rate: float, 
                     mutation_strength: float, rng):
    """Create mutations to fill population back to target size."""
    
    mutations = []
    next_id = max(agent.id for agent in survivors) + 1
    
    for agent in survivors:

        # Randomly decide whether to mutate this agent
        if rng.random() > mutation_rate:
            continue

        # Weight by fitness for parent selection
        weights = []
        for agent in survivors:
            # Get fitness from the most recent generation
            last_generation = max(agent.fitness_history.keys())
            fitness = agent.fitness_history[last_generation]
            weights.append(fitness)  # Ensure non-negative weights
        
        # Offset weights to avoid zero probabilities
        weight_offset = min(weights)
        weights = [(w - weight_offset+1e-10) for w in weights]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Select parent
        parent = rng.choice(survivors, p=weights)
        
        # Create mutation
        mutant = parent.mutate(next_id, generation, mutation_strength, rng)
        mutations.append(mutant)
        next_id += 1
    
    return mutations


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
    task_folder = args_cli.task.replace("-", "_").lower()

    # Setup logging
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "sb3", "population_evolution", task_folder, run_info)
    
    # Initialize evolution logger
    evolution_logger = EvolutionLogger(log_dir)
    
    evolution_logger.log("Population Evolution Training Started")
    evolution_logger.log(f"Population size: {args_cli.population_size}")
    evolution_logger.log(f"Generations: {args_cli.generations}")
    evolution_logger.log(f"Timesteps per agent: {args_cli.timesteps_per_agent}")
    evolution_logger.log(f"Mutation rate: {args_cli.mutation_rate}")
    evolution_logger.log(f"Mutation strength: {args_cli.mutation_strength}")
    evolution_logger.log(f"Selection randomness: {args_cli.selection_randomness}")
    evolution_logger.log(f"Logging to: {log_dir}")
    
    # Process agent config
    agent_cfg = process_sb3_cfg(agent_cfg)
    policy_arch = agent_cfg.pop("policy")
    agent_cfg.pop("n_timesteps", None)
    
    # Create initial population with random parameters
    total_ancestry = []
    population = []
    evolution_logger.log("Creating initial population:")

    if args_cli.disable_param_randomization:
        evolution_logger.log("  Initial parameter randomization disabled; using default parameters for all agents.")
        for i in range(args_cli.population_size):
            # Default parameters (e.g., classic claw gripper)
            default_length = (Agent.MIN_SCALE_LENGTH + Agent.MAX_SCALE_LENGTH) / 2
            default_width = (Agent.MIN_SCALE_WIDTH + Agent.MAX_SCALE_WIDTH) / 2
            rotations = np.linspace(0, 2*np.pi, num=Agent.NUM_PARAMS, endpoint=False)

            params = np.array([
                np.full(Agent.NUM_PARAMS, default_length),
                np.full(Agent.NUM_PARAMS, default_width),
                rotations
            ]).T

            agent = Agent(i, params, policy_arch, agent_cfg, generation=0)
            population.append(agent)
            formatted_params = np.round(params, 2).tolist()
            evolution_logger.log(f"Agent {i}: Params = {formatted_params}")

    else:
        for i in range(args_cli.population_size):
            # Generate random parameters
            mean_length = (Agent.MIN_SCALE_LENGTH + Agent.MAX_SCALE_LENGTH) / 2
            std_dev_length = (Agent.MAX_SCALE_LENGTH - Agent.MIN_SCALE_LENGTH) / 6  # 99.7% within 3 std deviations
            mean_width = (Agent.MIN_SCALE_WIDTH + Agent.MAX_SCALE_WIDTH) / 2
            std_dev_width = (Agent.MAX_SCALE_WIDTH - Agent.MIN_SCALE_WIDTH) / 6  # 99.7% within 3 std deviations

            rotations = np.linspace(0, 2*np.pi, num=Agent.NUM_PARAMS, endpoint=False)

            params = np.array([
                rng.normal(mean_length, std_dev_length, size=Agent.NUM_PARAMS),
                rng.normal(mean_width, std_dev_width, size=Agent.NUM_PARAMS),
                (rotations + rng.normal(0, np.pi/6, size=Agent.NUM_PARAMS)) % (2 * np.pi)
            ]).T

            agent = Agent(i, params, policy_arch, agent_cfg, generation=0)
            population.append(agent)
            formatted_params = np.round(params, 2).tolist()
            evolution_logger.log(f"Agent {i}: Params = {formatted_params}")

    total_ancestry += population

    # Evolution loop
    for generation in range(args_cli.generations):
        evolution_logger.log(f"{'='*50}")
        evolution_logger.log(f"GENERATION {generation + 1}/{args_cli.generations}")
        evolution_logger.log(f"{'='*50}")
        
        generation_fitness = []
        for agent in population:
            formatted_params = np.round(agent.params, 2).tolist()
            evolution_logger.log(f"Training Agent {agent.id} (params: {formatted_params})")

            # Set parameters and recreate environment
            evolution_logger.log("  Setting up environment...")
            modify_params_usd(task_folder, agent.params)
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
            fitness = agent.evaluate_fitness(env, generation)
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
        best_agent_id = population[np.argmax(generation_fitness)].id
        
        evolution_logger.log(f"Generation {generation + 1} Results:")
        evolution_logger.log(f"  Average fitness: {avg_fitness:.3f}")
        evolution_logger.log(f"  Best fitness: {best_fitness:.3f} (Agent {best_agent_id})")
        
        # On all but the last generation, mutate the population
        if generation < args_cli.generations-1:
        
            # Selection and mutation
            evolution_logger.log("  Performing selection and mutation...")
            
            # Select survivors
            survivors = select_survivors(
                population, 
                args_cli.population_size, 
                args_cli.selection_randomness, 
                rng
            )
            
            evolution_logger.log(f"  Selected {len(survivors)} survivors")
            
            # Create mutations
            mutations = mutate_population(
                survivors,
                generation + 1,
                args_cli.mutation_rate, 
                args_cli.mutation_strength,
                rng
            )
            
            evolution_logger.log(f"  Created {len(mutations)} mutations")
            
            # New population = survivors + mutations
            population = survivors + mutations
            total_ancestry += mutations

            # Delete old agent models to free memory
            used_ids = {agent.id for agent in population}
            for agent in total_ancestry:
                if agent.id not in used_ids and hasattr(agent, "model"):
                    del agent.model

            # Log family tree information
            originals = len([a for a in population if not a.family_tree])
            mutants = len([a for a in population if a.family_tree])
            evolution_logger.log(f"  Next generation: {originals} originals, {mutants} mutants")

    # Final report
    evolution_logger.log(f"{'='*50}")
    evolution_logger.log("EVOLUTION COMPLETE")
    evolution_logger.log(f"{'='*50}")
    
    # Find overall best agent across all generations
    all_fitness = []
    best_agents = []
    for agent in total_ancestry:
        max_fitness = max(agent.fitness_history.values())
        all_fitness.append(max_fitness)
        best_agents.append((agent, max_fitness))

    best_agent, best_fitness = max(best_agents, key=lambda x: x[1])
    
    evolution_logger.log(f"Best overall agent: {best_agent.id}")
    evolution_logger.log(f"Best fitness: {best_fitness:.3f}")
    evolution_logger.log(f"Best params: {best_agent.params}")
    best_generation = max(best_agent.fitness_history, key=lambda k: best_agent.fitness_history[k])
    evolution_logger.log(f"Generation: {best_generation+1}")
    evolution_logger.log(f"Family tree: {best_agent.family_tree}")
        
    # Log lineage information
    if best_agent.family_tree:
        evolution_logger.log(f"Lineage length: {len(best_agent.family_tree)} mutations")
        root_ancestor = best_agent.family_tree[0][1] if best_agent.family_tree else best_agent.id
        evolution_logger.log(f"Root ancestor: Agent {root_ancestor}")

    # Save population data with genetic algorithm information
    population_data = {
        "agents": [
            {
                "id": int(agent.id),
                "params": [[float(p) for p in row] for row in agent.params],
                "fitness_history": {int(gen): float(fitness) for gen, fitness in agent.fitness_history.items()},
                "best_fitness": float(max(agent.fitness_history.values())) if agent.fitness_history else 0.0,
                "generation": int(agent.generation),
                "parent_id": int(agent.parent_id) if agent.parent_id is not None else None,
                "family_tree": [(int(gen), int(parent)) for gen, parent in agent.family_tree]
            }
            for agent in total_ancestry
        ],
        "config": {
            "population_size": int(args_cli.population_size),
            "generations": int(args_cli.generations),
            "timesteps_per_agent": int(args_cli.timesteps_per_agent),
            "mutation_rate": float(args_cli.mutation_rate),
            "mutation_strength": float(args_cli.mutation_strength),
            "selection_randomness": float(args_cli.selection_randomness)
        },
        "evolution_stats": {
            "total_agents_created": len(total_ancestry),
            "original_agents": len([a for a in total_ancestry if not a.family_tree]),
            "mutated_agents": len([a for a in total_ancestry if a.family_tree]),
            "max_lineage_length": max(len(a.family_tree) for a in total_ancestry) if total_ancestry else 0
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

# Lineage Training and Visualization

This README explains how to use the lineage training and visualization scripts to analyze the evolution of specific agents.

## Overview

The lineage training system allows you to:
1. Take any agent from a population evolution run
2. Find its complete lineage (ancestor chain)
3. Train each step in the lineage for the full evolution timesteps
4. Visualize how much each mutation step could have improved with more training

## Scripts

### 1. `lineage_training.py`
Trains each step of an agent's lineage for the full evolution timesteps.

**Usage:**
```bash
cd IsaacLab
./isaaclab.bat -p ../scripts/lineage_training.py --results_file "path/to/population_results.yaml" --agent_id <agent_id>
```

**Arguments:**
- `--results_file`: Path to the `population_results.yaml` file from a population evolution run
- `--agent_id`: ID of the target agent whose lineage you want to train
- `--num_envs`: Number of environments (default: 1024)
- `--task`: Task name (default: "Grasp-Object")
- `--seed`: Random seed (default: 42)

**Example:**
```bash
cd IsaacLab
./isaaclab.bat -p ../scripts/lineage_training.py --results_file "logs/sb3/population_evolution/grasp_object/2025-08-07_14-30-15/population_results.yaml" --agent_id 25
```

### 2. `visualize_lineage.py`
Creates visualizations of the lineage training results.

**Usage:**
```bash
python scripts/visualize_lineage.py --results_file "path/to/lineage_training_results.yaml"
```

**Arguments:**
- `--results_file`: Path to the `lineage_training_results.yaml` file created by `lineage_training.py`
- `--output_dir`: Output directory for plots (optional, defaults to same directory as results file)

**Example:**
```bash
python scripts/visualize_lineage.py --results_file "logs/sb3/population_evolution/grasp_object/2025-08-07_14-30-15/lineage_training/agent_25/2025-08-07_15-45-30/lineage_training_results.yaml"
```

## Workflow

1. **Run Population Evolution:**
   ```bash
   cd IsaacLab
   ./isaaclab.bat -p ../scripts/population_evolution.py --population_size 8 --generations 5
   ```

2. **Find Your Target Agent:**
   Look at the evolution plots or logs to find an interesting agent ID (e.g., the best performing agent).

3. **Train the Lineage:**
   ```bash
   cd IsaacLab
   ./isaaclab.bat -p ../scripts/lineage_training.py --results_file "logs/sb3/population_evolution/grasp_object/[timestamp]/population_results.yaml" --agent_id [agent_id]
   ```

4. **Visualize Results:**
   ```bash
   python scripts/visualize_lineage.py --results_file "logs/sb3/population_evolution/grasp_object/[timestamp]/lineage_training/agent_[agent_id]/[timestamp]/lineage_training_results.yaml"
   ```

## Output Files

### Lineage Training Results
The `lineage_training.py` script creates:
- `lineage_training_results.yaml`: Complete results data
- Individual model checkpoints for each lineage step
- Training logs for each step

### Visualization Results
The `visualize_lineage.py` script creates:
- `lineage_analysis_agent_[id].png`: Main 4-panel analysis plot
- `lineage_summary_agent_[id].png`: Text summary with detailed results

## Visualizations Explained

The main analysis plot contains 4 panels:

1. **Original vs Trained Fitness**: Bar chart comparing the fitness each agent achieved during evolution vs. what it achieved with full training
2. **Fitness Improvement per Step**: Shows how much each step improved (green) or degraded (red) with additional training
3. **Lineage Performance Evolution**: Line plot showing the progression of both original and trained fitness along the lineage
4. **Cumulative Performance Metrics**: Shows cumulative improvement and maximum fitness achieved so far

## Use Cases

This analysis is useful for:
- **Understanding evolution dynamics**: See if early mutations were actually good but didn't get enough training time
- **Identifying promising lineages**: Find lineages where each step shows consistent improvement potential
- **Evaluating selection pressure**: See if the evolution correctly identified the best mutations
- **Research insights**: Understand how morphological changes affect learning potential vs. immediate performance

## Tips

1. **Choose interesting agents**: Look for agents that performed well, had interesting morphologies, or came from long lineages
2. **Compare multiple lineages**: Run this analysis on several agents to compare different evolutionary paths
3. **Consider computational cost**: Each lineage step requires full training, so longer lineages take more time
4. **Use adequate training time**: The training time is automatically set to `generations * timesteps_per_agent` from the original evolution

## Troubleshooting

- **Agent not found**: Make sure the agent ID exists in the population results file
- **Memory issues**: Reduce `--num_envs` if you run out of GPU memory
- **Import errors**: Make sure you're running from the correct environment with Isaac Lab activated

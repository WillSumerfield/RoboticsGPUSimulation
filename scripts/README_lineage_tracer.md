# Agent Lineage Tracer

This script traces the evolutionary family history of a specific agent and captures images of the robot morphology at each mutation step. It's perfect for visualizing how the robot's shape evolved through generations.

## Features

- **Complete Lineage Tracing**: Follows an agent's ancestry back to its root ancestor
- **Multi-Angle Photography**: Captures robot from multiple camera angles (1-8 angles)
- **High-Quality Images**: Configurable image resolution for detailed morphology analysis
- **Organized Output**: Creates structured folders with images and summary files
- **Family Tree Summary**: Generates text summary of the complete evolutionary path

## Usage

### Basic Usage
```bash
python scripts/trace_lineage.py --log_dir logs/sb3/population_evolution/2025-07-30_14-30-15 --agent_id 15
```

### Advanced Usage with Custom Settings
```bash
python scripts/trace_lineage.py \
    --log_dir logs/sb3/population_evolution/2025-07-30_14-30-15 \
    --agent_id 15 \
    --num_angles 8 \
    --image_size 1024 \
    --camera_distance 1.5 \
    --output_dir custom_lineage_images
```

## Arguments

- `--log_dir`: Path to your evolution training log directory (required)
- `--agent_id`: ID of the agent whose lineage you want to trace (required)
- `--task`: Task name (default: "Isaac-Isaacenv-Direct-v0")
- `--image_size`: Resolution of captured images in pixels (default: 512)
- `--camera_distance`: Distance of camera from robot (default: 2.0)
- `--num_angles`: Number of camera angles around the robot (default: 4, max: 8)
- `--output_dir`: Base directory for output images (default: "lineage_images")

## Output Structure

The script creates a timestamped directory structure:

```
lineage_images/
└── agent_15_lineage_20250802_143025/
    ├── lineage_summary.txt                    # Complete family tree summary
    ├── agent_3_gen_0/                         # Root ancestor
    │   ├── angle_00_params_1.20_0.85_1.65.png
    │   ├── angle_01_params_1.20_0.85_1.65.png
    │   ├── angle_02_params_1.20_0.85_1.65.png
    │   └── angle_03_params_1.20_0.85_1.65.png
    ├── agent_7_gen_3/                         # First mutation
    │   ├── angle_00_params_1.35_0.90_1.70.png
    │   └── ...
    └── agent_15_gen_7/                        # Target agent
        ├── angle_00_params_1.55_1.10_1.85.png
        └── ...
```

## Example Output

### Lineage Summary (lineage_summary.txt)
```
Agent Lineage Trace
Target Agent: 15
Total Generations: 4
Root Ancestor: Agent 3

Evolutionary Path:
Gen 0: Agent 3 [ROOT] - Params: (1.20, 0.85, 1.65) - Fitness: 0.234
Gen 3: Agent 7 ← Agent 3 - Params: (1.35, 0.90, 1.70) - Fitness: 0.456
Gen 5: Agent 12 ← Agent 7 - Params: (1.42, 0.95, 1.75) - Fitness: 0.578
Gen 7: Agent 15 ← Agent 12 - Params: (1.55, 1.10, 1.85) - Fitness: 0.723
```

## Camera Angles

The script captures images from evenly spaced angles around the robot:
- 1 angle: Front view
- 4 angles: Front, Right, Back, Left (90° apart)
- 8 angles: Every 45° around the robot

## Tips

1. **High-Resolution Images**: Use `--image_size 1024` or higher for detailed analysis
2. **Multiple Angles**: Use `--num_angles 8` for complete morphology documentation
3. **Close-Up Views**: Reduce `--camera_distance` to 1.0-1.5 for detailed digit structure
4. **Batch Processing**: You can run this script for multiple agents to compare lineages

## Integration with Evolution Training

This script works seamlessly with the population evolution training results. Just point it to any log directory from your evolution runs and specify any agent ID that appears in the results.

## Troubleshooting

- **Agent Not Found**: Make sure the agent ID exists in your population results
- **Missing Images**: Check that the USD file paths are correct for your environment
- **Low Quality Images**: Increase `--image_size` and adjust `--camera_distance`
- **Memory Issues**: Process one lineage at a time for very large family trees

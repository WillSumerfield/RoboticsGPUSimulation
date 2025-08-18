"""Lineage visualization script.

Reads lineage training results and creates visualizations showing
the performance evolution along the lineage.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from isaaclab.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Visualize lineage training results.")
    parser.add_argument("--results_file", type=str, required=True, 
                       help="Path to lineage_training_results.yaml file.")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots. If None, uses same dir as results file.")
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    #args.results_file = os.path.abspath(args.results_file)
    results = load_yaml(args.results_file)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.results_file)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    lineage_steps = results['lineage_steps']
    target_agent_id = results['target_agent_id']
    
    print(f"Visualizing lineage for agent {target_agent_id}")
    print(f"Lineage has {len(lineage_steps)} steps")
    
    # Create DataFrame for easier plotting
    df_data = []
    for step in lineage_steps:
        df_data.append({
            'Step': step['step_index'],
            'Agent_ID': step['agent_id'],
            'Generation': step['generation'],
            'Original_Fitness': step['original_fitness'],
            'Trained_Fitness': step['trained_fitness'],
            'Improvement': step['improvement'],
            'Additional_Timesteps': step.get('additional_timesteps', 0),
            'Checkpoint_Found': step.get('checkpoint_found', False)
        })
    
    df = pd.DataFrame(df_data)
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Lineage Training Analysis - Agent {target_agent_id}', 
                 fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Plot 1: Original vs Trained Fitness Comparison
    x_pos = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, df['Original_Fitness'], width, 
                    label='Original Fitness', color='#ff6b6b', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, df['Trained_Fitness'], width,
                    label='Trained Fitness', color='#4ecdc4', alpha=0.8)
    
    # Add value labels on bars
    for i, (orig, trained) in enumerate(zip(df['Original_Fitness'], df['Trained_Fitness'])):
        ax1.text(i - width/2, orig + max(df['Trained_Fitness']) * 0.01, f'{orig:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, trained + max(df['Trained_Fitness']) * 0.01, f'{trained:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_title('Original vs Trained Fitness', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lineage Step', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Step {i}\\nAgent {row.Agent_ID}' for i, row in df.iterrows()], 
                       rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitness Improvement per Step
    colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in df['Improvement']]
    bars = ax2.bar(df['Step'], df['Improvement'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, df['Improvement'])):
        height = bar.get_height()
        y_pos = height + (max(df['Improvement']) * 0.01) if height >= 0 else height - (max(df['Improvement']) * 0.01)
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos, f'{imp:+.2f}', 
                ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Fitness Improvement per Lineage Step', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lineage Step', fontsize=12)
    ax2.set_ylabel('Fitness Improvement', fontsize=12)
    ax2.set_xticks(df['Step'])
    ax2.set_xticklabels([f'Step {i}\\nAgent {row.Agent_ID}' for i, row in df.iterrows()], 
                       rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lineage Performance Evolution (Line Plot)
    ax3.plot(df['Step'], df['Original_Fitness'], 'o-', linewidth=3, markersize=8, 
            color='#ff6b6b', label='Original Fitness', alpha=0.8)
    ax3.plot(df['Step'], df['Trained_Fitness'], 's-', linewidth=3, markersize=8, 
            color='#4ecdc4', label='Trained Fitness', alpha=0.8)
    
    # Fill between for improvement visualization
    ax3.fill_between(df['Step'], df['Original_Fitness'], df['Trained_Fitness'], 
                    alpha=0.3, color='green', where=(df['Trained_Fitness'] >= df['Original_Fitness']),
                    interpolate=True, label='Improvement')
    ax3.fill_between(df['Step'], df['Original_Fitness'], df['Trained_Fitness'], 
                    alpha=0.3, color='red', where=(df['Trained_Fitness'] < df['Original_Fitness']),
                    interpolate=True, label='Degradation')
    
    ax3.set_title('Lineage Performance Evolution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Lineage Step', fontsize=12)
    ax3.set_ylabel('Fitness', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['Step'])
    
    # Plot 4: Cumulative Statistics
    cumulative_improvement = df['Improvement'].cumsum()
    max_fitness_so_far = df['Trained_Fitness'].cummax()
    
    ax4_twin = ax4.twinx()
    
    # Cumulative improvement (left y-axis)
    line1 = ax4.plot(df['Step'], cumulative_improvement, 'o-', linewidth=3, markersize=8, 
                    color='#9b59b6', label='Cumulative Improvement')
    ax4.set_ylabel('Cumulative Improvement', fontsize=12, color='#9b59b6')
    ax4.tick_params(axis='y', labelcolor='#9b59b6')
    
    # Maximum fitness achieved so far (right y-axis)
    line2 = ax4_twin.plot(df['Step'], max_fitness_so_far, 's-', linewidth=3, markersize=8, 
                         color='#f39c12', label='Max Fitness So Far')
    ax4_twin.set_ylabel('Maximum Fitness', fontsize=12, color='#f39c12')
    ax4_twin.tick_params(axis='y', labelcolor='#f39c12')
    
    ax4.set_title('Cumulative Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Lineage Step', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(df['Step'])
    
    # Add legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the main plot
    plot_path = os.path.join(output_dir, f"lineage_analysis_agent_{target_agent_id}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Main plot saved to: {plot_path}")
    
    # Create a summary statistics plot
    plt.figure(figsize=(12, 8))
    
    # Create summary statistics
    summary_stats = {
        'Total Steps': len(lineage_steps),
        'Best Original Fitness': df['Original_Fitness'].max(),
        'Best Trained Fitness': df['Trained_Fitness'].max(),
        'Average Improvement': df['Improvement'].mean(),
        'Total Cumulative Improvement': df['Improvement'].sum(),
        'Steps with Improvement': (df['Improvement'] > 0).sum(),
        'Steps with Degradation': (df['Improvement'] < 0).sum(),
        'Max Single Step Improvement': df['Improvement'].max(),
        'Max Single Step Degradation': df['Improvement'].min(),
        'Total Additional Timesteps': df['Additional_Timesteps'].sum(),
        'Checkpoints Found': df['Checkpoint_Found'].sum(),
        'Checkpoints Missing': len(df) - df['Checkpoint_Found'].sum(),
    }
    
    # Create text summary
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.title(f'Lineage Training Summary - Agent {target_agent_id}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Create summary text
    summary_text = f"""
    Lineage Analysis Results:
    ========================
    
    Target Agent: {target_agent_id}
    Total Lineage Steps: {summary_stats['Total Steps']}
    
    Checkpoint Recovery:
    -------------------
    Checkpoints Found: {summary_stats['Checkpoints Found']} / {summary_stats['Total Steps']}
    Checkpoints Missing: {summary_stats['Checkpoints Missing']}
    Total Additional Training: {summary_stats['Total Additional Timesteps']:,} timesteps
    
    Fitness Metrics:
    ----------------
    Best Original Fitness: {summary_stats['Best Original Fitness']:.3f}
    Best Trained Fitness: {summary_stats['Best Trained Fitness']:.3f}
    Overall Improvement: {summary_stats['Best Trained Fitness'] - summary_stats['Best Original Fitness']:+.3f}
    
    Step-by-Step Analysis:
    ----------------------
    Average Improvement per Step: {summary_stats['Average Improvement']:+.3f}
    Total Cumulative Improvement: {summary_stats['Total Cumulative Improvement']:+.3f}
    
    Steps with Improvement: {summary_stats['Steps with Improvement']} / {summary_stats['Total Steps']}
    Steps with Degradation: {summary_stats['Steps with Degradation']} / {summary_stats['Total Steps']}
    
    Best Single Step Improvement: {summary_stats['Max Single Step Improvement']:+.3f}
    Worst Single Step Degradation: {summary_stats['Max Single Step Degradation']:+.3f}
    
    Detailed Step Results:
    ----------------------
    """
    
    for _, row in df.iterrows():
        checkpoint_symbol = "✓" if row['Checkpoint_Found'] else "✗"
        summary_text += f"Step {row['Step']}: Agent {row['Agent_ID']} [{checkpoint_symbol}] (Gen {row['Generation']}) - "
        summary_text += f"Original: {row['Original_Fitness']:.3f}, Trained: {row['Trained_Fitness']:.3f}, "
        summary_text += f"Improvement: {row['Improvement']:+.3f}, Additional: {row['Additional_Timesteps']:,} steps\\n    "
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    summary_path = os.path.join(output_dir, f"lineage_summary_agent_{target_agent_id}.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Summary plot saved to: {summary_path}")
    
    plt.show()
    
    # Print console summary
    print(f"\\n{'='*60}")
    print("LINEAGE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Target Agent: {target_agent_id}")
    print(f"Lineage Length: {summary_stats['Total Steps']} steps")
    print(f"Checkpoints Found: {summary_stats['Checkpoints Found']}/{summary_stats['Total Steps']}")
    print(f"Total Additional Training: {summary_stats['Total Additional Timesteps']:,} timesteps")
    print(f"Best Original Fitness: {summary_stats['Best Original Fitness']:.3f}")
    print(f"Best Trained Fitness: {summary_stats['Best Trained Fitness']:.3f}")
    print(f"Overall Improvement: {summary_stats['Best Trained Fitness'] - summary_stats['Best Original Fitness']:+.3f}")
    print(f"Average Improvement per Step: {summary_stats['Average Improvement']:+.3f}")
    print(f"Steps with Improvement: {summary_stats['Steps with Improvement']}/{summary_stats['Total Steps']}")


if __name__ == "__main__":
    main()

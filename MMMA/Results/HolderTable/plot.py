import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from cycler import cycler

# Function to read and process a single file
def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    metrics = []
    for experiment in data:
        experiment_metrics = [experiment[i] for i in range(4)]  # First 4 metrics
        metrics.append(experiment_metrics)
    return np.array(metrics)

# Read all files
file_pattern = '*.npy'
file_paths = glob.glob(file_pattern)

# Process all files
all_data = [process_file(file_path) for file_path in file_paths]

# Calculate median across experiments for each file
median_data = [np.median(file_data, axis=0) for file_data in all_data]

# Convert to numpy array for easier indexing
median_data = np.array(median_data)

# Plotting
metric_names = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
iterations = range(1, len(median_data[0][0]) + 1)

# Create a directory to save plots if it doesn't exist
output_dir = 'metric_plots'
os.makedirs(output_dir, exist_ok=True)

# Generate file names for the legend, truncated to 10 characters
file_names = [os.path.basename(file_path)[:16] + '...' for file_path in file_paths]

# Define a fixed color cycle
colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(file_paths)))
plt.rc('axes', prop_cycle=(cycler('color', colors)))

for i, metric in enumerate(metric_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for j, (file_data, file_name) in enumerate(zip(all_data, file_names)):
        color = colors[j]
        
        # Plot individual runs with transparency
        for run in file_data:
            ax.plot(iterations, run[i], color=color, alpha=0.1, linewidth=0.5)
        
        # Plot median
        ax.plot(iterations, np.median(file_data, axis=0)[i], color=color, label=file_name, linewidth=2)

    if metric in ["Simple Regret", "Cumulative Regret"]:
        ax.set_title(f'Log {metric} Across 10 Experiments')
        ax.set_yscale('log')
    else:
        ax.set_title(f'Median {metric} Across 10 Experiments')
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Value')
    
    # Adjust legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', framealpha=0.5)
    
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Plots have been saved in the '{output_dir}' directory.")
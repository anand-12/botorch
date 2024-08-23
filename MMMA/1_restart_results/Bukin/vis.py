import numpy as np
import matplotlib.pyplot as plt
import glob
import os

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

# Generate file names for the legend
file_names = [os.path.basename(file_path) for file_path in file_paths]

for i, metric in enumerate(metric_names):
    if metric == "Simple Regret" or metric == "Cumulative Regret":
        plt.figure(figsize=(10, 6))
        for j, file_data in enumerate(median_data):
            plt.plot(iterations, file_data[i], label=file_names[j])
        plt.title(f'Log {metric} Across 10 Experiments')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        for j, file_data in enumerate(median_data):
            plt.plot(iterations, file_data[i], label=file_names[j])
        plt.title(f'Median {metric} Across 10 Experiments')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"Plots have been saved in the '{output_dir}' directory.")
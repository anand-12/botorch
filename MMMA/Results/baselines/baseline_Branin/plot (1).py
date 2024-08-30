import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Function to read and process a single file
def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    metrics = []
    for experiment in data:
        experiment_metrics = [experiment[i][:100] for i in range(4)]  # First 4 metrics, first 100 iterations
        metrics.append(experiment_metrics)
    return np.array(metrics)

# Function to extract kernel and acquisition function from filename
def extract_info(filename):
    parts = filename.split('_')
    kernel = next(part for part in parts if part in ['Matern52', 'RBF'])
    acquisition = next(part for part in parts if part in ['LogEI', 'LogPI', 'UCB'])
    return kernel, acquisition

# Read all files
file_pattern = '*.npy'
file_paths = glob.glob(file_pattern)

# Process all files
all_data = [process_file(file_path) for file_path in file_paths]

# Plotting
metric_names = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
iterations = range(1, 101)  # 100 iterations

# Create a directory to save plots if it doesn't exist
output_dir = 'metric_plots'
os.makedirs(output_dir, exist_ok=True)

# Define colors and line styles
colors = {'Matern52': 'blue', 'RBF': 'red'}
line_styles = {'LogEI': '-', 'LogPI': '--', 'UCB': ':'}

for i, metric in enumerate(metric_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for file_path, file_data in zip(file_paths, all_data):
        kernel, acquisition = extract_info(os.path.basename(file_path))
        color = colors[kernel]
        line_style = line_styles[acquisition]
        
        # Calculate median and percentiles
        median = np.median(file_data[:, i, :], axis=0)
        percentile_25 = np.percentile(file_data[:, i, :], 25, axis=0)
        percentile_75 = np.percentile(file_data[:, i, :], 75, axis=0)
        
        # Plot median
        label = f"{kernel} - {acquisition}"
        ax.plot(iterations, median, color=color, linestyle=line_style, label=label, linewidth=2)
        
        # Plot shaded area for 25-75 percentiles
        ax.fill_between(iterations, percentile_25, percentile_75, color=color, alpha=0.1)

    if metric in ["Simple Regret", "Cumulative Regret"]:
        ax.set_title(f'{metric} for Branin in log scale')
        ax.set_yscale('log')
    else:
        ax.set_title(f'Median {metric} for Branin in log scale')
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel(metric)
    
    # Adjust legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', framealpha=0.5)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Plots have been saved in the '{output_dir}' directory.")
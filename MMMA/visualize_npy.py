import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def plot_results():
    # Get all .npy files in the current directory
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]
    
    # Group results by experiment type
    results = defaultdict(list)
    for file in npy_files:
        experiment_type = file.split('_')[0]  # Assumes the first part of the filename is the experiment type
        results[experiment_type].append(np.load(file, allow_pickle=True))
    
    metrics = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(12, 8))
        
        for (experiment_type, experiment_results), color in zip(results.items(), colors):
            for result in experiment_results:
                plt.plot(result[i], alpha=0.1, color=color)
            
            median = np.median([result[i] for result in experiment_results], axis=0)
            plt.plot(median, color=color, linewidth=2, label=f'{experiment_type} Median')
        
        plt.xlabel("Iterations")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Iterations")
        plt.legend()
        
        # Add a text box with final values
        textstr = "Final Values:\n"
        for experiment_type, experiment_results in results.items():
            final_value = np.median([result[i][-1] for result in experiment_results])
            textstr += f'{experiment_type}: {final_value:.4f}\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f"{metric.replace(' ', '_').lower()}_comparison.png")
        plt.close()
    
    print("Plots saved as PNG files.")

# Usage
plot_results()
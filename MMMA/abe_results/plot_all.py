import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from matplotlib.colors import rgb2hex

def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    metrics = []
    for experiment in data:
        experiment_metrics = [experiment[i] for i in range(4)]  
        metrics.append(experiment_metrics)
    return np.array(metrics)

def get_function_name(file_name):
    return file_name.split('_')[0]

def plot_metrics(file_paths, output_dir):
    all_data = [process_file(file_path) for file_path in file_paths]
    file_names = [os.path.basename(file_path).replace('.npy', '') for file_path in file_paths]
    metric_names = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    
    # Get unique function names and assign colors
    function_names = list(set(get_function_name(name) for name in file_names))
    color_palette = sns.color_palette("husl", n_colors=len(function_names))
    color_dict = {func: rgb2hex(color) for func, color in zip(function_names, color_palette)}
    
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for i, metric in enumerate(metric_names):
        fig, ax = plt.subplots(figsize=(12, 8))
        for file_data, file_name in zip(all_data, file_names):
            iterations = range(1, len(file_data[0][0]) + 1)
            median = np.median(file_data, axis=0)[i]
            lower = np.percentile(file_data, 25, axis=0)[i]
            upper = np.percentile(file_data, 75, axis=0)[i]
            
            function_name = get_function_name(file_name)
            color = color_dict[function_name]
            
            # Determine line style
            if "least_risk" in file_name:
                linestyle = '--'
                label = f"{file_name} (Least Risk)"
            else:
                linestyle = '-'
                label = file_name
            
            ax.plot(iterations, median, linewidth=2, label=label, color=color, linestyle=linestyle)
            ax.fill_between(iterations, lower, upper, alpha=0.2, color=color)

        ax.set_ylabel(metric, fontsize=14)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='small', loc='center left', 
                   bbox_to_anchor=(1, 0.5), ncol=1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def process_folder(folder_path):
    file_pattern = os.path.join(folder_path, '*.npy')
    file_paths = glob.glob(file_pattern)
    
    if file_paths:
        output_dir = os.path.join(folder_path, 'metric_plots')
        plot_metrics(file_paths, output_dir)
        print(f"Plots have been saved in the '{output_dir}' directory.")
    else:
        print(f"No .npy files found in {folder_path}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing .npy files: ")
    process_folder(folder_path)
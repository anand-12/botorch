import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from matplotlib.lines import Line2D

# Define an extended color palette
color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#41f221', '#00CED1', '#0000CD', '#8B4513', '#FF1493', '#00FF00', '#FF4500', '#1E90FF', '#009999']

method_styles = {
    'base': {'color': color_palette[0]},
    'GPHedge_bandit': {'color': color_palette[1]},
    'GPHedge_uniform': {'color': color_palette[2]},
    'GPHedge_abe': {'color': color_palette[3]},
    'MMMA_BMA_bandit': {'color': color_palette[4]},
    'MMMA_uniform_bandit': {'color': color_palette[5]},
    'MMMA_uniform_random': {'color': color_palette[6]},
    'MMMA_BMA_random': {'color': color_palette[7]},
    'MultiModel_BMA': {'color': color_palette[8]},
    'MultiModel_uniform': {'color': color_palette[9]},
    'MultiModel_abe': {'color': color_palette[10]},
    'MMMA_abe_bandit': {'color': color_palette[11]},
    'MMMA_abe_random': {'color': color_palette[12]},
    'MMMA_BMA_abe': {'color': color_palette[13]},
    'MMMA_uniform_abe': {'color': color_palette[14]},
    'MMMA_abe_abe': {'color': color_palette[15]},
    'GPHedgeyq_bandit': {'color': color_palette[16]}
}

# List of folders to ignore
IGNORE_FOLDERS = ['Result_Images', 'baselines']  # Add folders you want to ignore here

def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    metrics = []
    for experiment in data:
        experiment_metrics = [experiment[i] for i in range(4)]  
        metrics.append(experiment_metrics)
    return np.array(metrics)

def create_adaptive_legend(ax, lines, labels, position):
    legend_elements = []
    for line, label in zip(lines, labels):
        color = line.get_color()
        legend_elements.append(Line2D([0], [0], color=color, lw=4, label=label))
    
    bbox = ax.get_position()
    if position == 'top':
        legend_bbox = (bbox.x0, bbox.y1, bbox.width, 0.1)
        loc = 'lower left'
    else:  
        legend_bbox = (bbox.x0, bbox.y0, bbox.width, 0.1)
        loc = 'upper left'
    
    ax.legend(handles=legend_elements, loc=loc, bbox_to_anchor=legend_bbox,
              ncol=3, mode="expand", borderaxespad=0., fontsize='small')

def plot_metrics(file_paths, output_dir, dataset_name):
    all_data = [process_file(file_path) for file_path in file_paths]
    file_names = [os.path.basename(file_path).replace('.npy', '') for file_path in file_paths]
    metric_names = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    iterations = range(1, len(all_data[0][0][0]) + 1)

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for i, metric in enumerate(metric_names):
        fig, ax = plt.subplots(figsize=(10, 7))
        lines = []
        for file_data, file_name in zip(all_data, file_names):
            style = method_styles.get(file_name, {'color': 'gray'})
            median = np.median(file_data, axis=0)[i]
            lower = np.percentile(file_data, 25, axis=0)[i]
            upper = np.percentile(file_data, 75, axis=0)[i]
            line, = ax.plot(iterations, median, color=style['color'], linewidth=2)
            ax.fill_between(iterations, lower, upper, color=style['color'], alpha=0.2)
            lines.append(line)

        if metric in ["Simple Regret"] and dataset_name != 'Shekel':
            ax.set_yscale('log')
            plt.subplots_adjust(bottom=0.05)  
            legend_position = 'top'
        else:
            legend_position = 'bottom'
            plt.subplots_adjust(bottom=0.05)  # Make room for legend at the bottom

        ax.set_ylabel(metric, fontsize=18, fontweight='bold')  # Increase y-axis label size and make it bold

        # Increase the font size and weight of the tick labels
        ax.tick_params(axis='both', which='major', labelsize=18, width=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        
        create_adaptive_legend(ax, lines, file_names, legend_position)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}_{dataset_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots for {dataset_name} have been saved in the '{output_dir}' directory.")

def process_datasets(base_path):
    for folder in os.listdir(base_path):
        if folder in IGNORE_FOLDERS:
            print(f"Ignoring folder: {folder}")
            continue
        
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            file_pattern = os.path.join(folder_path, '*.npy')
            all_file_paths = glob.glob(file_pattern)
            
            # Filter out 'PI_is_back' files and include only the specified file names
            valid_file_names = [
                'base', 'GPHedge_bandit', 'GPHedge_uniform', 'GPHedge_abe',
                'MMMA_BMA_bandit', 'MMMA_uniform_bandit', 'MMMA_uniform_random',
                'MMMA_BMA_random', 'MultiModel_BMA', 'MultiModel_uniform', 'MultiModel_abe',
                'MMMA_abe_bandit', 'MMMA_abe_random', 'MMMA_BMA_abe', 'MMMA_uniform_abe', 'MMMA_abe_abe', 'GPHedgeyq_bandit'
            ]
            file_paths = [fp for fp in all_file_paths if any(vfn in fp for vfn in valid_file_names)]
            
            if file_paths:
                output_dir = os.path.join(base_path, 'metric_plots')
                plot_metrics(file_paths, output_dir, folder)
            else:
                print(f"No valid .npy files found in {folder}")

if __name__ == "__main__":
    base_path = '.'  
    process_datasets(base_path)
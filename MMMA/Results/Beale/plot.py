import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from matplotlib.lines import Line2D

# Define a distinct color palette
color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

method_styles = {
    'base': {'color': color_palette[0]},
    'GPHedge_bandit': {'color': color_palette[1]},
    'GPHedge_uniform': {'color': color_palette[2]},
    'MMMA_BMA_bandit': {'color': color_palette[3]},
    'MMMA_uniform_bandit': {'color': color_palette[4]},
    'MMMA_uniform_random': {'color': color_palette[5]},
    'MultiModel_BMA': {'color': color_palette[6]},
    'MultiModel_uniform': {'color': color_palette[7]}
}

def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    metrics = []
    for experiment in data:
        experiment_metrics = [experiment[i] for i in range(4)]  # First 4 metrics
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
    else:  # bottom
        legend_bbox = (bbox.x0, bbox.y0, bbox.width, 0.1)
        loc = 'upper left'
    
    ax.legend(handles=legend_elements, loc=loc, bbox_to_anchor=legend_bbox,
              ncol=4, mode="expand", borderaxespad=0., fontsize='small')

def plot_metrics(file_paths, output_dir):
    all_data = [process_file(file_path) for file_path in file_paths]
    file_names = [os.path.basename(file_path).replace('.npy', '') for file_path in file_paths]

    metric_names = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    iterations = range(1, len(all_data[0][0][0]) + 1)

    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for i, metric in enumerate(metric_names):
        print(metric)
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

        if metric in ["Simple Regret"]:
            ax.set_title(f'Log {metric}')
            ax.set_yscale('log')
            legend_position = 'top'
            plt.subplots_adjust(top=0.95)  # Make room for legend at the top
        else:
            ax.set_title(f'{metric}')
            legend_position = 'bottom'
            plt.subplots_adjust(bottom=0.05)  # Make room for legend at the bottom
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Value')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create and place the adaptive legend
        create_adaptive_legend(ax, lines, file_names, legend_position)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots have been saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    file_pattern = '*.npy'
    file_paths = glob.glob(file_pattern)
    output_dir = 'metric_plots'
    plot_metrics(file_paths, output_dir)
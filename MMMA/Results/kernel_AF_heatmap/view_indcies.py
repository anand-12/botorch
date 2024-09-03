import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(folder_path):
    data = {}
    method_order = ['base', 'MultiModel_likelihood', 'MultiModel_uniform', 
                    'GPHedge_bandit', 'GPHedge_random',
                    'MMMA_likelihood_bandit', 'MMMA_likelihood_random', 
                    'MMMA_uniform_bandit', 'MMMA_uniform_random']
    
    for method_name in method_order:
        file_path = os.path.join(folder_path, f"{method_name}.npy")
        if os.path.exists(file_path):
            method_data = np.load(file_path, allow_pickle=True)
            
            if method_name == 'base':
                kernel_selections = method_data[0][-2]
                af_selections = method_data[0][-1]
            elif method_name.startswith('MultiModel'):
                kernel_selections = method_data[0][-2]
                af_selections = method_data[0][-1]
            elif method_name.startswith('GPHedge'):
                kernel_selections = method_data[0][-1]
                af_selections = method_data[0][-2]
            elif method_name.startswith('MMMA'):
                kernel_selections = method_data[0][-1]
                af_selections = method_data[0][-2]
            
            data[method_name] = {
                'kernels': kernel_selections,
                'acquisitions': af_selections
            }
    
    return data

def plot_custom_rocket_style_heatmap(data, selection_type, options, custom_labels):
    fig, ax = plt.subplots(figsize=(15, 8)) 

    groups = ['base', 'MultiModel', 'GPHedge', 'MMMA']

    y_positions = []
    y_ticks = []

    y_pos = 0 
    for group_idx, group in enumerate(groups):
        group_methods = [method for method in data.keys() if method.startswith(group)]

        for method_idx, method in enumerate(group_methods):
            selections = data[method][selection_type]
            heatmap_data = np.array([options.index(s) if s in options else -1 for s in selections])
            if selection_type == "kernels":
                scatter = ax.scatter(range(len(selections)), [y_pos] * len(selections), 
                                    c=heatmap_data, cmap=plt.cm.get_cmap('Set1', len(options)), 
                                    marker='s', s=250, vmin=-0.5, vmax=len(options)-0.5)  # Larger square boxes
            else:
                scatter = ax.scatter(range(len(selections)), [y_pos] * len(selections), 
                                    c=heatmap_data, cmap=plt.cm.get_cmap('summer', len(options)), 
                                    marker='s', s=250, vmin=-0.5, vmax=len(options)-0.5)
            ax.text(-5, y_pos, custom_labels.get(method, method), ha='right', va='center', fontsize=18, fontweight='bold')
            
            y_positions.append(y_pos)
            y_ticks.append(y_pos)

            y_pos -= 1  
        ax.axvline(x=-2.75, color='black', linewidth=2)
        if group_idx < len(groups) - 1:
            ax.axhline(y=y_pos + 0.5, color='black', linewidth=2)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-6, len(selections) + 2)
    ax.set_ylim(y_pos - 1, 1) 

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks(range(len(options)))
    cbar.set_ticklabels(options)
    cbar.ax.tick_params(labelsize=25)

    plt.tight_layout(pad=1)  

    function_name = "Hartmann"
    filename = f"{function_name}_{selection_type}.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {filename}")

folder_path = './Hartmann_2'
data = load_data(folder_path)

kernel_options = ['Matern52', 'Matern32', 'RBF']
af_options = ['LogEI', 'LogPI', 'UCB']

custom_labels = {
    'base': 'Standard\nBO',
    'MultiModel_likelihood': 'MM\nBMA',
    'MultiModel_uniform': 'MM\nRand',
    'GPHedge_bandit': 'MA\nBandit',
    'GPHedge_random': 'MA\nRand',
    'MMMA_likelihood_bandit': 'MMMA\nBMA-Bandit',
    'MMMA_likelihood_random': 'MMMA\nBMA-Rand',
    'MMMA_uniform_bandit': 'MMMA\nRand-Bandit',
    'MMMA_uniform_random': 'MMMA\nRand-Rand'
}

# Plot kernel selections
plot_custom_rocket_style_heatmap(data, 'kernels', kernel_options, custom_labels)

# Plot acquisition function selections
plot_custom_rocket_style_heatmap(data, 'acquisitions', af_options, custom_labels)
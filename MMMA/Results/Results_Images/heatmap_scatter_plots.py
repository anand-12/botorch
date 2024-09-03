import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as stats
import textwrap

def process_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        cumulative_regrets = [experiment[3] for experiment in data]  # Index 3 for cumulative regret
        return cumulative_regrets
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def power_law(x, a, b):
    return a * np.power(x, b)

def fit_sublinearity(x, y):
    popt, _ = curve_fit(power_law, x, y)
    return popt[1]  # Return the exponent

def analyze_regret(regrets):
    sublinearity_orders = []
    for regret in regrets:
        x = np.arange(1, len(regret) + 1)
        sublinearity_order = fit_sublinearity(x, regret)
        sublinearity_orders.append(sublinearity_order)

    mean_sublinearity = np.mean(sublinearity_orders)
    std_error = stats.sem(sublinearity_orders)

    return mean_sublinearity, std_error

def plot_relative_heatmap(df_mean, x_label_map, show_numbers=True):
    df_relative = df_mean.div(df_mean.min(axis=1), axis=0) - 1

    plt.figure(figsize=(20, 16))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 16

    cmap = sns.color_palette("flare", as_cmap=True)
    vmax = np.percentile(df_relative.values, 90)

    if show_numbers:
        sns.heatmap(df_relative, annot=df_mean.round(3), fmt='.3f', cmap=cmap, 
                    cbar_kws={'label': 'Relative difference from best'},
                    vmin=0, vmax=vmax, annot_kws={"size": 14, "weight": "bold"})
    else:
        sns.heatmap(df_relative, annot=False, cmap=cmap, 
                    cbar_kws={'label': 'Relative difference from best'},
                    vmin=0, vmax=vmax)

    # Use the x_label_map directly, preserving line breaks
    x_labels = [x_label_map.get(col, col) for col in df_mean.columns]
    plt.gca().set_xticklabels(x_labels, rotation=0, ha='center', fontsize=16, fontweight='bold')

    plt.yticks(fontsize=16, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')

    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Relative difference from best', fontsize=16, fontweight='bold')

    plt.tight_layout()
    filename = "relative_sublinearity_heatmap_with_numbers.png" if show_numbers else "relative_sublinearity_heatmap_without_numbers.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_function_scatter(folder, function_results, x_label_map):
    function_name = os.path.basename(folder)
    plt.figure(figsize=(16, 10))
    methods = list(function_results.keys())
    means = [function_results[method]['mean'] for method in methods]
    sems = [function_results[method]['sem'] for method in methods]

    # Use custom x-labels and wrap them
    x_labels = [x_label_map.get(method, method) for method in methods]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=10)) for label in x_labels]

    # Set up color palette with three distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Create scatter plot with error bars
    for i, (method, mean, sem) in enumerate(zip(methods, means, sems)):
        if i == 0:
            color = colors[0]  # First color for the first method
        elif 1 <= i <= 4:
            color = colors[1]  # Second color for the next 5 methods
        else:
            color = colors[2]  # Third color for the last 3 methods
        plt.errorbar(i, mean, yerr=sem, fmt='o', capsize=5, capthick=2, color=color, ecolor=color, markersize=10, alpha=0.7, elinewidth=2)

    # Customize the plot
    plt.ylabel("Sublinearity Order", fontsize=18, fontweight='bold')
    plt.title(f"Sublinearity Orders for {function_name}", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize x-axis with wrapped labels
    plt.xticks(range(len(methods)), wrapped_labels, rotation=0, ha='center', fontsize=12)

    # Add horizontal lines for visual reference
    plt.axhline(y=np.mean(means), color='r', linestyle='--', alpha=0.5)
    plt.text(len(methods)-1, np.mean(means), 'Mean', va='center', ha='left', backgroundcolor='w', fontsize=10)

    # Add vertical lines to separate method groups
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for wrapped labels
    plt.savefig(f"sublinearity_scatter_{function_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_folders(folders):
    methods = ['Base', 'MultiModel_BMA', 'MultiModel_uniform', 'GPHedge_bandit', 
               'GPHedge_random', 'MMMA_BMA_random', 'MMMA_BMA_bandit', 'MMMA_uniform_bandit', 'MMMA_uniform_random']

    x_label_map = {
        'Base': 'Standard\nBO',
        'MultiModel_BMA': 'MM\nBMA',
        'MultiModel_uniform': 'MM\nRand',
        'GPHedge_bandit': 'MA\nBandit',
        'GPHedge_random': 'MA\nRand',
        'MMMA_BMA_random': 'MMMA\nBMA\nRand',
        'MMMA_BMA_bandit': 'MMMA\nBMA\nBandit',
        'MMMA_uniform_bandit': 'MMMA\nRand\nBandit',
        'MMMA_uniform_random': 'MMMA\nRand\nRand'
    }

    results = {folder: {method: {'mean': 0, 'sem': 0} for method in methods} for folder in folders}

    for folder in folders:
        for method in methods:
            file_path = os.path.join(folder, f'{method}.npy')
            if os.path.exists(file_path):
                regrets = process_file(file_path)
                if regrets:
                    mean_sublinearity, std_error = analyze_regret(regrets)
                    results[folder][method]['mean'] = mean_sublinearity
                    results[folder][method]['sem'] = std_error

        # Create individual scatter plot for each function
        plot_function_scatter(folder, results[folder], x_label_map)

    df_mean = pd.DataFrame({method: [results[folder][method]['mean'] for folder in folders] 
                            for method in methods}, index=[os.path.basename(folder) for folder in folders])

    plot_relative_heatmap(df_mean, x_label_map, show_numbers=True)
    plot_relative_heatmap(df_mean, x_label_map, show_numbers=False)

if __name__ == "__main__":
    folders = ['./../Ackley', 
               './../Beale', 
               './../Branin', 
               './../Bukin', 
               './../Cosine8', 
               './../DixonPrice', 
               './../DropWave', 
               './../Griewank', 
               './../Hartmann', 
               './../Levy', 
               './../Michalewicz', 
               './../Rastrigin', 
               './../Rosenbrock', 
               './../SixHumpCamel', 
               './../ThreeHumpCamel', 
               './../Shekel']
    process_folders(folders)
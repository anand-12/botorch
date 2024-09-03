import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def process_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        final_gap_metrics = [experiment[1][-1] for experiment in data]
        return np.array(final_gap_metrics)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return np.array([])

def plot_ensemble_gap_metric(folders):
    data_dict = {
        'Base': [],
        'MultiModel_BMA': [],
        'MultiModel_uniform': [],
        'GPHedge_bandit': [],
        'GPHedge_random': [],
        'MMMA_BMA_random': [],
        'MMMA_BMA_bandit': [],
        'MMMA_uniform_bandit': [],
        'MMMA_uniform_random': []
    }

    for folder in folders:
        for method in data_dict.keys():
            if method == 'Base':
                file_path = os.path.join(folder, 'Base.npy')
            else:
                file_path = os.path.join(folder, f'{method}.npy')

            if os.path.exists(file_path):
                data_dict[method].extend(process_file(file_path))

    df = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})

    # Define custom y-axis labels (previously x-axis)
    y_label_map = {
        'Base': 'Standard\nBO',
        'MultiModel_BMA': 'MM-BMA',
        'MultiModel_uniform': 'MM-Rand',
        'GPHedge_bandit': 'MA\nBandit',
        'GPHedge_random': 'MA\nRand',
        'MMMA_BMA_random': 'MMMA\nBMA\nRand',
        'MMMA_BMA_bandit': 'MMMA\nBMA\nBandit',
        'MMMA_uniform_bandit': 'MMMA\nRand\nBandit',
        'MMMA_uniform_random': 'MMMA\nRand\nRand'
    }

    plt.figure(figsize=(8, 12))  # Adjusted figure size for vertical orientation
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # More pleasing color palette
    n_colors = len(data_dict)
    palette = sns.color_palette("rocket", n_colors)
    sns.set_palette(palette)

    ax = sns.boxplot(data=df, orient='h', showfliers=False, width=0.6)

    plt.xlabel("Final Gap Metric", fontsize=22, fontweight='bold')
    plt.ylabel("")

    # Apply custom y-axis labels
    ax.set_yticklabels([y_label_map.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()], 
                       rotation=0, va='center', fontsize=18, fontweight='bold')

    # Adjust y-axis label positions
    for tick in ax.get_yticklabels():
        tick.set_x(-0.01)

    plt.axhline(y=0.5, color='#777777', linestyle='--', alpha=0.7, linewidth=2)
    plt.axhline(y=4.5, color='#777777', linestyle='--', alpha=0.7, linewidth=2)

    # Center the text labels and adjust their horizontal position
    plt.text(plt.xlim()[1] * 1.02, 0, 'Baseline', va='center', ha='left', fontsize=20, fontweight='bold', color='#555555', rotation=-90)
    plt.text(plt.xlim()[1] * 1.02, 2, 'Single Ensemble', va='center', ha='left', fontsize=20, fontweight='bold', color='#555555', rotation=-90)
    plt.text(plt.xlim()[1] * 1.02, 7, 'Multi-level Ensemble', va='baseline', ha='left', fontsize=20, fontweight='bold', color='#555555', rotation=-90)

    ax.tick_params(axis='x', labelsize=18, labelcolor='#555555')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('ensemble_gap_metric_boxplot_rotated.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Rotated box plot has been saved as 'ensemble_gap_metric_boxplot_rotated.png'.")

if __name__ == "__main__":
    folders = ['./../Ackley', './../Beale', './../Branin', './../Bukin', './../Cosine8', './../DixonPrice', './../DropWave', './../Griewank', 
               './../Hartmann', './../Levy', './../Michalewicz', './../Rastrigin', './../Rosenbrock', './../SixHumpCamel', './../ThreeHumpCamel', './../Shekel']
    plot_ensemble_gap_metric(folders)
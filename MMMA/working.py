import botorch
import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.kernels import RFFKernel, SpectralDeltaKernel
from botorch.test_functions.synthetic import Hartmann, Ackley, Levy, StyblinskiTang
import warnings
import time
from utils import *
from plotting import *
from botorch_test_functions import setup_test_function, true_maxima
import random
import os


botorch.settings.debug = True

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_function_name = "Hartmann" 

test_function, bounds = setup_test_function(test_function_name)
true_max = true_maxima[test_function_name]
target_func = target_function(test_function)

n_iterations = 100
kernel_types_list = [['Matern52', 'RFF', 'Matern32'], ['Matern52', 'RFF', 'Matern32']]
acq_func_types = [['logEI'], ['logEI', 'UCB', 'PI']]

num_exp = len(kernel_types_list)

titles = [f"{kernel_types_list[i]} {acq_func_types[i]}" for i in range(num_exp)]

n_seeds = 10
all_seeds_results = []

for seed in range(3225, 3225+n_seeds):
    print(f"\nRunning experiment with seed {seed}")
    
    for i in range(num_exp):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        init_x, init_y, best_init_y = generate_initial_data(10, n_dim=bounds.size(1), test_func=target_func)
        initial_data = {
            "train_x": init_x.to(device),
            "train_y": init_y.to(device),
            "best_init_y": best_init_y,
            "bounds": bounds,
            "true_maximum": true_max
        }
        s = time.time()
        result = run_experiment(n_iterations, kernel_types_list[i], acq_func_types[i], initial_data, target_func)
        print(f"Time taken for {titles[i]}: {time.time() - s:.2f} seconds")
        all_seeds_results.append(result)


def plot_results(all_seeds_results, n_iterations, num_exp, titles, true_maximum, test_function_name):
    output_path = f"{test_function_name}"
    os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist
    
    metrics = ['Best Observed Values', 'Simple Regrets', 'Cumulative Regrets']
    metric_indices = [0, 6, 7]  # Indices for best_observed_values, simple_regrets, cumulative_regrets
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        plt.title(f'Bayesian Optimization Results - {metric} ({test_function_name})', fontsize=16)
        
        for exp in range(num_exp):
            exp_data = [all_seeds_results[exp + j*num_exp][metric_indices[i]] for j in range(n_seeds)]
            median_data = np.median(exp_data, axis=0)
            
            # Plot individual seed results as light dashed lines
            for seed_data in exp_data:
                plt.plot(range(1, len(seed_data) + 1), seed_data, linestyle='--', alpha=0.3, color=f'C{exp}')
            
            # Plot median as solid line
            plt.plot(range(1, len(median_data) + 1), median_data, label=titles[exp], linewidth=2)
        
        plt.xlabel('Iterations')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        
        # Mark true maximum for Best Observed Values plot
        if i == 0:
            plt.axhline(y=true_maximum, color='r', linestyle='--', label='True Maximum')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/{test_function_name}_{metric.replace(" ", "_").lower()}.png')
        plt.close()

# Usage
true_maximum = true_max  # Use the true_max variable from your code
plot_results(all_seeds_results, n_iterations, num_exp, titles, true_maximum, test_function_name)

# median_best_values = []
# median_simple_regrets = []
# median_cumulative_regrets = []
# median_gap_metrics = []
# median_execution_times = []

# for i in range(num_exp):  
#     best_values = np.median([seed_results[i][0] for seed_results in all_seeds_results], axis=0)
#     simple_regrets = np.median([seed_results[i][6] for seed_results in all_seeds_results], axis=0)
#     cumulative_regrets = np.median([seed_results[i][7] for seed_results in all_seeds_results], axis=0)
#     gap_metrics = np.median([seed_results[i][4] for seed_results in all_seeds_results], axis=0)
#     execution_time = np.median([seed_results[i][5] for seed_results in all_seeds_results])
    
#     median_best_values.append(best_values)
#     median_simple_regrets.append(simple_regrets)
#     median_cumulative_regrets.append(cumulative_regrets)
#     median_gap_metrics.append(gap_metrics)
#     median_execution_times.append(execution_time)


# plot_results([(bv, None, None, true_max, gm, et, sr, cr) 
#               for bv, gm, et, sr, cr in zip(median_best_values, median_gap_metrics, median_execution_times, median_simple_regrets, median_cumulative_regrets)], 
#              titles, f"{str(test_function)}_median", true_max)


# plot_regret_growth(all_seeds_results, titles, str(test_function))


# plot_all_cumulative_regrets(all_seeds_results, median_cumulative_regrets, titles, str(test_function))


# for i in range(num_exp):
#     print(f"\nMedian Regret Analysis for {titles[i]}:")
#     print(f"Final Simple Regret: {median_simple_regrets[i][-1]:.4f}")
#     print(f"Final Cumulative Regret: {median_cumulative_regrets[i][-1]:.4f}")
#     print(f"Median Simple Regret: {np.median(median_simple_regrets[i]):.4f}")
#     print(f"Regret Reduction: {(median_simple_regrets[i][0] - median_simple_regrets[i][-1]) / median_simple_regrets[i][0] * 100:.2f}%")


# save_results(all_seeds_results, str(test_function))
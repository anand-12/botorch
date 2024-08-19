import numpy as np
import matplotlib.pyplot as plt

import pickle


def save_results(all_seeds_results, test_function_name):
    with open(f"{test_function_name}_all_seeds_results.pkl", "wb") as f:
        pickle.dump(all_seeds_results, f)
    print(f"Results saved to {test_function_name}_all_seeds_results.pkl")


def load_results(test_function_name):
    with open(f"{test_function_name}_all_seeds_results.pkl", "rb") as f:
        return pickle.load(f)

def average_results(all_results):
    avg_results = []
    for i in range(len(all_results[0])):
        avg_best_observed = np.mean([res[i][0] for res in all_results], axis=0)
        avg_gap_metrics = np.mean([res[i][4] for res in all_results], axis=0)
        avg_execution_time = np.mean([res[i][5] for res in all_results])
        avg_results.append((avg_best_observed, None, all_results[0][i][3], avg_gap_metrics, avg_execution_time))
    return avg_results

def median_results(all_results):
    median_results = []
    for i in range(len(all_results[0])):
        median_best_observed = np.median([res[i][0] for res in all_results], axis=0)
        median_gap_metrics = np.median([res[i][4] for res in all_results], axis=0)
        median_execution_time = np.median([res[i][5] for res in all_results])
        median_results.append((median_best_observed, None, all_results[0][i][3], median_gap_metrics, median_execution_time))
    return median_results

def plot_average_results(avg_results, titles, test_function_name, true_maximum):
    fig, axs = plt.subplots(len(avg_results), 1, figsize=(10, 6*len(avg_results)))
    
    if len(avg_results) == 1:
        axs = [axs]  

    for i, result in enumerate(avg_results):
        avg_best_observed, _, _, _, avg_gap_metrics, _ = result
        
        axs[i].plot(avg_best_observed, marker='o', linestyle='-', color='b', label='Avg Best Objective Value')
        axs[i].axhline(y=true_maximum, color='k', linestyle='--', label='True Maxima')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Average Best Objective Function Value")
        axs[i].legend()
        axs[i].grid(True)

        ax2 = axs[i].twinx()
        ax2.plot(avg_gap_metrics, marker='x', linestyle='-', color='r', label='Avg Gap Metric')
        ax2.set_ylabel("Average Gap Metric G_i")
        ax2.legend(loc='lower right')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"avg_test_res_{test_function_name}.png")
    plt.close()

def plot_average_execution_times(avg_execution_times, titles, test_function_name):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(avg_execution_times)), avg_execution_times)
    plt.xlabel("Experiment")
    plt.ylabel("Average Execution Time (seconds)")
    plt.title(f"Average Execution Times for {test_function_name}")
    plt.xticks(range(len(avg_execution_times)), [f"Exp {i+1}" for i in range(len(avg_execution_times))], rotation=45)
    for i, v in enumerate(avg_execution_times):
        plt.text(i, v, f"{v:.2f}s", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"avg_execution_times_{test_function_name}.png")
    plt.close()

def plot_average_final_gap_metrics(avg_gap_metrics, titles, test_function_name):
    plt.figure(figsize=(10, 6))
    final_gap_metrics = [gap_metric[-1] for gap_metric in avg_gap_metrics]
    plt.bar(range(len(final_gap_metrics)), final_gap_metrics)
    plt.xlabel("Experiment")
    plt.ylabel("Average Final Gap Metric")
    plt.title(f"Average Final Gap Metrics for {test_function_name}")
    plt.xticks(range(len(final_gap_metrics)), [f"Exp {i+1}" for i in range(len(final_gap_metrics))], rotation=45)
    for i, v in enumerate(final_gap_metrics):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"avg_final_gap_metrics_{test_function_name}.png")
    plt.close()

def plot_median_results(median_results, titles, test_function_name, true_maximum):
    fig, axs = plt.subplots(len(median_results), 1, figsize=(10, 6*len(median_results)))
    
    if len(median_results) == 1:
        axs = [axs]  

    for i, result in enumerate(median_results):
        median_best_observed, _, _, _, median_gap_metrics, _ = result
        
        axs[i].plot(median_best_observed, marker='o', linestyle='-', color='b', label='Median Best Objective Value')
        axs[i].axhline(y=true_maximum, color='k', linestyle='--', label='True Maxima')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Median Best Objective Function Value")
        axs[i].legend()
        axs[i].grid(True)

        ax2 = axs[i].twinx()
        ax2.plot(median_gap_metrics, marker='x', linestyle='-', color='r', label='Median Gap Metric')
        ax2.set_ylabel("Median Gap Metric G_i")
        ax2.legend(loc='lower right')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"median_test_res_{test_function_name}.png")
    plt.close()

def plot_median_execution_times(median_execution_times, titles, test_function_name):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(median_execution_times)), median_execution_times)
    plt.xlabel("Experiment")
    plt.ylabel("Median Execution Time (seconds)")
    plt.title(f"Median Execution Times for {test_function_name}")
    plt.xticks(range(len(median_execution_times)), [f"Exp {i+1}" for i in range(len(median_execution_times))], rotation=45)
    for i, v in enumerate(median_execution_times):
        plt.text(i, v, f"{v:.2f}s", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"median_execution_times_{test_function_name}.png")
    plt.close()

def plot_median_final_gap_metrics(median_gap_metrics, titles, test_function_name):
    plt.figure(figsize=(10, 6))
    final_gap_metrics = [gap_metric[-1] for gap_metric in median_gap_metrics]
    plt.bar(range(len(final_gap_metrics)), final_gap_metrics)
    plt.xlabel("Experiment")
    plt.ylabel("Median Final Gap Metric")
    plt.title(f"Median Final Gap Metrics for {test_function_name}")
    plt.xticks(range(len(final_gap_metrics)), [f"Exp {i+1}" for i in range(len(final_gap_metrics))], rotation=45)
    for i, v in enumerate(final_gap_metrics):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"median_final_gap_metrics_{test_function_name}.png")
    plt.close()

def plot_execution_times(execution_times, titles, test_function_name):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(execution_times)), execution_times)
    plt.xlabel("Experiment")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Execution Times for {test_function_name}")
    plt.xticks(range(len(execution_times)), [f"Exp {i+1}" for i in range(len(execution_times))], rotation=45)
    for i, v in enumerate(execution_times):
        plt.text(i, v, f"{v:.2f}s", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"execution_times_{test_function_name}.png")
    

def plot_final_gap_metrics(gap_metrics, titles, test_function_name):
    plt.figure(figsize=(10, 6))
    final_gap_metrics = [gap_metric[-1] for gap_metric in gap_metrics]
    plt.bar(range(len(final_gap_metrics)), final_gap_metrics)
    plt.xlabel("Experiment")
    plt.ylabel("Final Gap Metric")
    plt.title(f"Final Gap Metrics for {test_function_name}")
    plt.xticks(range(len(final_gap_metrics)), [f"Exp {i+1}" for i in range(len(final_gap_metrics))], rotation=45)
    for i, v in enumerate(final_gap_metrics):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"final_gap_metrics_{test_function_name}.png")
    
def plot_results(results, titles, test_function_name, true_maximum):
    fig, axs = plt.subplots(len(results), 1, figsize=(12, 6*len(results)))
    if len(results) == 1:
        axs = [axs]  

    for i, (best_observed_values, chosen_acq_functions, selected_models, _, 
            gap_metrics, execution_time, simple_regrets, cumulative_regrets) in enumerate(results):
        
        if np.isscalar(best_observed_values):
            axs[i].axhline(y=best_observed_values, color='b', linestyle='-', label='Best Objective Value')
            final_best = best_observed_values
        else:
            axs[i].plot(best_observed_values, marker='o', linestyle='-', color='b', label='Best Objective Value')
            final_best = best_observed_values[-1]

        axs[i].axhline(y=true_maximum, color='k', linestyle='--', label='True Maximum')
        axs[i].set_title(f"{titles[i]}\nExecution Time: {execution_time:.2f}s")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Best Objective Function Value")
        axs[i].legend(loc='upper left')
        axs[i].grid(True)

        ax2 = axs[i].twinx()
        if np.isscalar(gap_metrics):
            ax2.axhline(y=gap_metrics, color='r', linestyle='-', label='Gap Metric')
            final_gap = gap_metrics
        else:
            ax2.plot(gap_metrics, marker='x', linestyle='-', color='r', label='Gap Metric')
            final_gap = gap_metrics[-1]

        ax2.set_ylabel("Gap Metric G_i")
        ax2.legend(loc='lower right')

        ax2.set_ylim(bottom=0)

        axs[i].text(0.02, 0.98, f"Final Best: {final_best:.4f}\nFinal Gap: {final_gap:.4f}", 
                    transform=axs[i].transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"test_res_{test_function_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_all_cumulative_regrets(all_seeds_results, median_cumulative_regrets, titles, test_function_name):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(titles)))
    
    for i, color in enumerate(colors):
        for seed_results in all_seeds_results:
            plt.plot(seed_results[i][7], alpha=0.3, color=color, linestyle='-', linewidth=0.5)
        plt.plot(median_cumulative_regrets[i], label=titles[i], linewidth=2, color=color)

    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Cumulative Regret for All Runs - {test_function_name}")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  
    plt.savefig(f"all_runs_cumulative_regret_{test_function_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_regret_growth(all_seeds_results, titles, test_function_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(titles)))

    for i, color in enumerate(colors):
        simple_regrets = [seed_result[i][6] for seed_result in all_seeds_results]
        cumulative_regrets = [seed_result[i][7] for seed_result in all_seeds_results]

        # Plot simple regrets
        if np.isscalar(simple_regrets[0]):
            ax1.axhline(y=simple_regrets[0], color=color, label=f"{titles[i]} (Scalar)")
        else:
            for regret in simple_regrets:
                ax1.plot(regret, alpha=0.3, color=color, linewidth=0.5)
            median_simple_regret = np.median(simple_regrets, axis=0)
            ax1.plot(median_simple_regret, label=titles[i], linewidth=2, color=color)

        # Plot cumulative regrets
        if np.isscalar(cumulative_regrets[0]):
            ax2.axhline(y=cumulative_regrets[0], color=color, label=f"{titles[i]} (Scalar)")
        else:
            for regret in cumulative_regrets:
                ax2.plot(regret, alpha=0.3, color=color, linewidth=0.5)
            median_cumulative_regret = np.median(cumulative_regrets, axis=0)
            ax2.plot(median_cumulative_regret, label=titles[i], linewidth=2, color=color)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Simple Regret")
    ax1.set_title(f"Simple Regret Growth for {test_function_name}")
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')

    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title(f"Cumulative Regret Growth for {test_function_name}")
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"regret_growth_{test_function_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
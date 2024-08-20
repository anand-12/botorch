import numpy as np
import matplotlib.pyplot as plt

def plot_results(baseline_file, portfolio_file, true_ensemble_file, false_ensemble_file, true_MMMA_file, false_MMMA_file):
    baseline_results = np.load(baseline_file, allow_pickle=True)
    portfolio_results = np.load(portfolio_file, allow_pickle=True)
    true_ensemble = np.load(true_ensemble_file, allow_pickle=True)
    false_ensemble = np.load(false_ensemble_file, allow_pickle=True)
    true_MMMA = np.load(true_MMMA_file, allow_pickle=True)
    false_MMMA = np.load(false_MMMA_file, allow_pickle=True)

    metrics = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        
        # Plot baseline results
        for result in baseline_results:
            plt.plot(result[i], alpha=0.1, color='blue')
        baseline_median = np.median([result[i] for result in baseline_results], axis=0)
        plt.plot(baseline_median, 'b-', linewidth=2, label='Baseline Median')
        
        # Plot portfolio results
        for result in portfolio_results:
            plt.plot(result[i], alpha=0.1, color='red')
        portfolio_median = np.median([result[i] for result in portfolio_results], axis=0)
        plt.plot(portfolio_median, 'r-', linewidth=2, label='Portfolio Median')

        # Plot true ensemble results
        for result in true_ensemble:
            plt.plot(result[i], alpha=0.1, color='green')
        true_ensemble_median = np.median([result[i] for result in true_ensemble], axis=0)
        plt.plot(true_ensemble_median, 'g-', linewidth=2, label='True Ensemble Median')

        # Plot false ensemble results
        for result in false_ensemble:
            plt.plot(result[i], alpha=0.1, color='purple')
        false_ensemble_median = np.median([result[i] for result in false_ensemble], axis=0)
        plt.plot(false_ensemble_median, 'm-', linewidth=2, label='False Ensemble Median')

        # Plot true MMMA results
        for result in true_MMMA:
            plt.plot(result[i], alpha=0.1, color='orange')
        true_MMMA_median = np.median([result[i] for result in true_MMMA], axis=0)
        plt.plot(true_MMMA_median, 'y-', linewidth=2, label='True MMMA Median')

        # Plot false MMMA results
        for result in false_MMMA:
            plt.plot(result[i], alpha=0.1, color='black')
        false_MMMA_median = np.median([result[i] for result in false_MMMA], axis=0)
        plt.plot(false_MMMA_median, 'k-', linewidth=2, label='False MMMA Median')

        
        plt.xlabel("Iterations")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Iterations")
        plt.legend()
        
        # Add a text box with final values
        final_baseline = baseline_median[-1]
        final_portfolio = portfolio_median[-1]
        final_true_ensemble = true_ensemble_median[-1]
        final_false_ensemble = false_ensemble_median[-1]
        textstr = f'Final Values:\nBaseline: {final_baseline:.4f}\nPortfolio: {final_portfolio:.4f}\nTrue Ensemble: {final_true_ensemble:.4f}\nFalse Ensemble: {final_false_ensemble:.4f}\n True MMMA: {false_MMMA_median[-1]:.4f}\nFalse MMMA: {false_MMMA_median[-1]:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f"{metric.replace(' ', '_').lower()}_comparison.png")
        plt.close()

    print("Plots saved as PNG files.")

# Usage
plot_results('baseline_Hartmann_optimization_results.npy', 'portfolio_Hartmann_optimization_results.npy', 'True_ensemble_Hartmann_optimization_results.npy', 'False_ensemble_Hartmann_optimization_results.npy', 'True_MMMA_Hartmann_optimization_results.npy', 'False_MMMA_Hartmann_optimization_results.npy')
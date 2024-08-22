import numpy as np
import matplotlib.pyplot as plt
import os

def parse_filename(filename):
    parts = filename.replace('.npy', '').split('_')
    exp_type = '_'.join(parts[:parts.index('function')])
    function = parts[parts.index('function') + 1]
    print()
    kernels = parts[parts.index('kernel') + 1:parts.index('acquisition')]
    acquisitions = parts[parts.index('acquisition') + 1:-2]
    return exp_type, function, kernels, acquisitions

def plot_results():
    function_name = str(input("Enter the function name of files to search: "))
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy') and function_name in f]
    
    results = {}
    for file in npy_files:
        exp_type, function, kernels, acquisitions = parse_filename(file)
        print(function)
        data = np.load(file, allow_pickle=True)
        key = f"{exp_type}_{'+'.join(kernels)}_{'+'.join(acquisitions)}"
        results[key] = np.mean(data, axis=0)  # Take median across experiments
    
    metrics = ["Maximum Value", "Gap Metric", "Simple Regret", "Cumulative Regret"]
    
    # Define colors for different experiment types
    exp_colors = {
        'baseline': 'blue',
        'portfolio': 'red',
        'True_uniform_MMMA': 'green',
        'False_uniform_MMMA': 'orange',
        'True_likelihood_MMMA': 'purple',
        'False_likelihood_MMMA': 'brown',
        'True_uniform_ensemble': 'pink',
        'False_uniform_ensemble': 'gray',
        'True_likelihood_ensemble': 'olive',
        'False_likelihood_ensemble': 'cyan'
    }
    
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(12, 8))
        
        for exp_name, exp_results in results.items():
            exp_type = exp_name.split('_')[0]
            if exp_type in ['True', 'False']:
                exp_type = '_'.join(exp_name.split('_')[:3])
            color = exp_colors.get(exp_type, 'gray')
            
            plt.plot(exp_results[i], color=color, label=exp_name)
        
        plt.xlabel("Iterations")
        plt.ylabel(metric)
        plt.title(f"{function} - {metric} vs Iterations")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{function}_{metric.replace(' ', '_').lower()}.png", bbox_inches='tight')
        plt.close()
        
    print("All plots saved as PNG files.")

# Usage
plot_results()
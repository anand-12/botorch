import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.optim import optimize_acqf
from botorch_test_functions import setup_test_function, true_maxima
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RFFKernel
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def gap_metric(f_start, f_current, f_star):
    return np.abs((f_start - f_current) / (f_start - f_star))

def get_next_points(train_X, train_Y, best_train_Y, bounds, kernel, iteration, total_iterations, switch_percentage):
    base_kernel = {
        'Matern52': MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
        'RBF': RBFKernel(ard_num_dims=train_X.shape[-1]),
        'Matern32': MaternKernel(nu=1.5, ard_num_dims=train_X.shape[-1]),
        'RFF': RFFKernel(num_samples=1000, ard_num_dims=train_X.shape[-1])
    }[kernel]
    
    single_model = SingleTaskGP(train_X, train_Y, covar_module=ScaleKernel(base_kernel))
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)

    if iteration < (switch_percentage / 100) * total_iterations:
        acq_function = LogExpectedImprovement(model=single_model, best_f=best_train_Y)
        acq_name = 'LogEI'
    else:
        acq_function = LogProbabilityOfImprovement(model=single_model, best_f=best_train_Y)
        acq_name = 'LogPI'

    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200}
    )

    return candidates, acq_name, single_model

def run_experiment(args):
    objective, bounds = setup_test_function(args.function, dim=args.dim)
    bounds = bounds.to(dtype=dtype, device=device)
    
    train_X = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
    train_Y = -objective(train_X).unsqueeze(-1)
    best_init_y = train_Y.max().item()
    best_train_Y = best_init_y
    
    true_max = true_maxima[args.function]
    
    max_values = [best_train_Y]
    gap_metrics = [gap_metric(best_init_y, best_init_y, true_max)]
    simple_regrets = [true_max - best_train_Y]
    cumulative_regrets = [true_max - best_train_Y]
    chosen_acq_functions = []

    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}, Best value = {best_train_Y:.4f}")
        new_candidates, acq_name, single_model = get_next_points(
            train_X, train_Y, best_train_Y, bounds, args.kernel, i, args.iterations, args.switch_percentage
        )
        new_Y = -objective(new_candidates).unsqueeze(-1)

        train_X = torch.cat([train_X, new_candidates])
        train_Y = torch.cat([train_Y, new_Y])
        best_train_Y = train_Y.max().item()
        
        max_values.append(best_train_Y)
        gap_metrics.append(gap_metric(best_init_y, best_train_Y, true_max))
        simple_regrets.append(true_max - best_train_Y)
        cumulative_regrets.append(cumulative_regrets[-1] + (true_max - new_Y.item()))
        chosen_acq_functions.append(acq_name)

    return max_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_acq_functions

def plot_results(all_best_observed, all_chosen_acq, args):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, best_observed in enumerate(all_best_observed):
        plt.plot(best_observed, alpha=0.3)
    median_best_observed = np.median(all_best_observed, axis=0)
    plt.plot(median_best_observed, color='red', linewidth=2, label='Median')
    plt.axhline(y=true_maxima[args.function], color='green', linestyle='--', label='True Maximum')
    plt.title(f"Best Objective Function Value for {-1*args.function}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.legend()

    plt.subplot(1, 2, 2)
    acq_counts = {'LogEI': [], 'LogPI': []}
    for i in range(args.iterations):
        logei_count = sum(1 for exp in all_chosen_acq if exp[i] == 'LogEI')
        logpi_count = sum(1 for exp in all_chosen_acq if exp[i] == 'LogPI')
        acq_counts['LogEI'].append(logei_count / args.experiments * 100)
        acq_counts['LogPI'].append(logpi_count / args.experiments * 100)

    plt.plot(acq_counts['LogEI'], label='LogEI')
    plt.plot(acq_counts['LogPI'], label='LogPI')
    plt.title("Acquisition Function Usage")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage Used")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization with LogEI to LogPI switch')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--kernel', type=str, default='Matern52', 
                        choices=['Matern52', 'RBF', 'Matern32', 'RFF'],
                        help='GP kernel to use')
    parser.add_argument('--experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--function', type=str, default='Hartmann', choices=list(true_maxima.keys()),
                        help='Test function to optimize')
    parser.add_argument('--dim', type=int, default=6, help='Dimensionality of the problem (for functions that support variable dimensions)')
    parser.add_argument('--switch_percentage', type=float, default=50, help='Percentage of iterations after which to switch from LogEI to LogPI')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_results = []

    for i in range(args.experiments):
        print(f"Running experiment {i+1}/{args.experiments}")
        experiment_results = run_experiment(args)
        all_results.append(experiment_results)

    all_results_np = np.array(all_results, dtype=object)
    np.save(f"logEI_logPI_switch_{args.function}_optimization_results.npy", all_results_np)

    print(f"Results saved to logEI_logPI_switch_{args.function}_optimization_results.npy")

    all_best_observed = [result[0] for result in all_results]
    all_chosen_acq = [result[4] for result in all_results]
    plot_results(all_best_observed, all_chosen_acq, args)
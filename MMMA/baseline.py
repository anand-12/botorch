import torch
import argparse
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import PosteriorMean, LogProbabilityOfImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RFFKernel
from botorch_test_functions import setup_test_function, true_maxima
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize, standardize
import warnings
import random
import os
import time
import gpytorch

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def gap_metric(f_start, f_current, f_star):
    return np.abs((f_start - f_current) / (f_start - f_star))

def get_kernel(kernel_name, dim):
    base_kernels = {
        'Matern52': MaternKernel(nu=2.5, ard_num_dims=dim),
        'Matern32': MaternKernel(nu=1.5, ard_num_dims=dim),
        'RBF': RBFKernel(ard_num_dims=dim),
        'RFF': RFFKernel(num_samples=100, ard_num_dims=dim)
    }
    return ScaleKernel(base_kernels[kernel_name])

def bayesian_optimization(n_iterations, seed, acq_func_name, kernel_name, test_func_name, dim):
    objective, bounds = setup_test_function(test_func_name, dim)
    bounds = bounds.to(dtype=dtype, device=device)
    f_star = true_maxima[test_func_name]
    num_initial_points = int(0.1 * n_iterations)
    train_X = draw_sobol_samples(bounds=bounds, n=num_initial_points, q=1).squeeze(1)
    train_Y = objective(train_X).unsqueeze(-1)
    
    best_observed_value = train_Y.max().item()
    f_start = best_observed_value
    
    best_observed_values = [best_observed_value]
    gap_metrics = [gap_metric(f_start, best_observed_value, f_star)]
    simple_regrets = [f_star - best_observed_value]
    cumulative_regrets = [f_star - best_observed_value]
    
    for iteration in range(n_iterations):
        # Compute bounds for normalization
        fit_bounds = torch.stack([torch.min(train_X, 0)[0], torch.max(train_X, 0)[0]])
        
        # Normalize inputs and standardize outputs
        train_X_normalized = normalize(train_X, bounds=fit_bounds)
        train_Y_standardized = standardize(train_Y)

        model = SingleTaskGP(train_X_normalized, train_Y_standardized, covar_module=get_kernel(kernel_name, dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.cholesky_jitter(1e-1):
            fit_gpytorch_mll(mll)
        
        # Standardize best observed value
        best_f = (best_observed_value - train_Y.mean()) / train_Y.std()

        acq_func = {
            'EI': ExpectedImprovement(model=model, best_f=best_f),
            'LogEI': LogExpectedImprovement(model=model, best_f=best_f),
            'PI': ProbabilityOfImprovement(model=model, best_f=best_f),
            'LogPI': LogProbabilityOfImprovement(model=model, best_f=best_f),
            'UCB': UpperConfidenceBound(model=model, beta=0.1),
            'PM': PosteriorMean(model=model)
        }[acq_func_name]
        
        # Optimize acquisition function in normalized space
        new_x_normalized, _ = optimize_acqf(
            acq_function=acq_func, 
            bounds=normalize(bounds, fit_bounds),
            q=1, 
            num_restarts=2, 
            raw_samples=50
        )
        
        # Unnormalize new_x before evaluating objective
        new_x = unnormalize(new_x_normalized, bounds=fit_bounds)
        new_y = objective(new_x).unsqueeze(-1)
        
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])
        
        best_observed_value = train_Y.max().item()
        
        best_observed_values.append(best_observed_value)
        gap_metrics.append(gap_metric(f_start, best_observed_value, f_star))
        simple_regrets.append(f_star - best_observed_value)
        cumulative_regrets.append(cumulative_regrets[-1] + (f_star - best_observed_value))
        
        print(f"Iteration {iteration + 1:>2}: Best value = {best_observed_value:.4f}")

    return best_observed_values, gap_metrics, simple_regrets, cumulative_regrets

def run_experiments(args):
    all_results = []
    n_iterations = 100
    for seed in range(args.seed, args.seed+args.experiments):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"\nExperiment Seed: {seed})")
        start_time = time.time()
        best_values, gap_metrics, simple_regrets, cumulative_regrets = bayesian_optimization(n_iterations, seed, args.acquisition, args.kernel, args.function, args.dim)
        end_time = time.time()
        experiment_time = end_time - start_time
        all_results.append([best_values, gap_metrics, simple_regrets, cumulative_regrets, experiment_time])
        print(f"Best value: {best_values[-1]:.4f}")
        print(f"Experiment time for baseline: {experiment_time:.2f} seconds")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization on various test functions')
    parser.add_argument('--experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--acquisition', type=str, default='LogEI', choices=['EI', 'LogEI', 'PI', 'LogPI', 'UCB', 'PM'], 
                        help='Acquisition function (EI: Expected Improvement, LogEI: Log Expected Improvement, PI: Probability of Improvement, LogPI, UCB: Upper Confidence Bound, PM: Posterior Mean)')
    parser.add_argument('--kernel', type=str, default='Matern52', choices=['Matern52', 'Matern32', 'RBF', 'RFF'], 
                        help='GP kernel (Matern52: Matérn 5/2 kernel, Matern32: Matérn 3/2 kernel, RBF: Radial Basis Function kernel, RFF: Random Fourier Features kernel)')
    parser.add_argument('--function', type=str, default='Hartmann', choices=list(true_maxima.keys()),
                        help='Test function to optimize')
    parser.add_argument('--dim', type=int, default=6, help='Dimension for functions that support variable dimensions')
    
    args = parser.parse_args()
    
    all_results = run_experiments(args)

    # Save results as .npy file
    all_results_np = np.array(all_results, dtype=object)
    os.makedirs(f"./{args.function}", exist_ok=True)
    np.save(f"./{args.function}/baseline_function_{args.function}{args.dim}_kernel_{args.kernel}_acquisition_{args.acquisition}_optimization_results.npy", all_results_np)
    print(f"Results saved to baseline_function_{args.function}{args.dim}_kernel_{args.kernel}_acquisition_{args.acquisition}_optimization_results.npy")
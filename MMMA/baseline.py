import torch
import argparse
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch_test_functions import setup_test_function, true_maxima
from botorch.utils.sampling import draw_sobol_samples
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def bayesian_optimization(n_iterations, seed, acq_func_name, kernel, test_func_name, dim):
    torch.manual_seed(seed)
    objective, bounds = setup_test_function(test_func_name, dim)
    bounds = bounds.to(dtype=dtype, device=device)

    train_X = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
    train_Y = -objective(train_X).unsqueeze(-1)
    
    best_observed_value = train_Y.max().item()
    
    for iteration in range(n_iterations):
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=dim) if kernel == 'matern' else RBFKernel(ard_num_dims=dim)
        model = SingleTaskGP(train_X, train_Y, covar_module=ScaleKernel(base_kernel))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        acq_func = {
            'ei': ExpectedImprovement(model=model, best_f=best_observed_value),
            'logei': LogExpectedImprovement(model=model, best_f=best_observed_value),
            'poi': ProbabilityOfImprovement(model=model, best_f=best_observed_value),
            'ucb': UpperConfidenceBound(model=model, beta=0.1)
        }[acq_func_name]
        
        new_x, _ = optimize_acqf(acq_function=acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
        new_y = -objective(new_x).unsqueeze(-1)
        
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])
        
        best_observed_value = train_Y.max().item()
        
        print(f"Iteration {iteration + 1:>2}: Best value = {best_observed_value:.4f}")

    return best_observed_value

def run_experiments(n_iterations, n_experiments, base_seed, acq_func_name, kernel, test_func_name, dim):
    results = []
    for i in range(n_experiments):
        seed = base_seed + i
        print(f"\nExperiment {i+1}/{n_experiments} (Seed: {seed})")
        best_value = bayesian_optimization(n_iterations, seed, acq_func_name, kernel, test_func_name, dim)
        results.append(best_value)
        print(f"Best value: {best_value:.4f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization on various test functions')
    parser.add_argument('--iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--acquisition', type=str, default='ei', choices=['ei', 'logei', 'poi', 'ucb'], 
                        help='Acquisition function (ei: Expected Improvement, logei: Log Expected Improvement, poi: Probability of Improvement, ucb: Upper Confidence Bound)')
    parser.add_argument('--kernel', type=str, default='matern', choices=['matern', 'rbf'], 
                        help='GP kernel (matern: Matérn kernel, rbf: Radial Basis Function kernel)')
    parser.add_argument('--function', type=str, default='Hartmann', choices=list(true_maxima.keys()),
                        help='Test function to optimize')
    parser.add_argument('--dim', type=int, default=2, help='Dimension for functions that support variable dimensions')
    
    args = parser.parse_args()
    
    results = run_experiments(args.iterations, args.experiments, args.seed, args.acquisition, args.kernel, args.function, args.dim)
    
    print("\nSummary of results:")
    print(f"Test function: {args.function}")
    print(f"True maximum: {true_maxima[args.function]}")
    print(f"Mean best value: {np.mean(results):.4f}")
    print(f"Std dev of best values: {np.std(results):.4f}")
    print(f"Min best value: {np.min(results):.4f}")
    print(f"Max best value: {np.max(results):.4f}")
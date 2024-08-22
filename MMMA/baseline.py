import torch
import argparse
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, LinearKernel, PolynomialKernel, ScaleKernel, RFFKernel
from botorch_test_functions import setup_test_function, true_maxima
from botorch.utils.sampling import draw_sobol_samples
import warnings, random
from botorch.acquisition.analytic import LogProbabilityOfImprovement


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
        model = SingleTaskGP(train_X, train_Y, covar_module=get_kernel(kernel_name, dim))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        acq_func = {
            'EI': ExpectedImprovement(model=model, best_f=best_observed_value),
            'LogEI': LogExpectedImprovement(model=model, best_f=best_observed_value),
            'PI': ProbabilityOfImprovement(model=model, best_f=best_observed_value),
            'LogPI': LogProbabilityOfImprovement(model=model, best_f=best_observed_value),
            'UCB': UpperConfidenceBound(model=model, beta=0.1),
            'PM': PosteriorMean(model=model)
        }[acq_func_name]
        
        new_x, _ = optimize_acqf(acq_function=acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
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
    n_iterations = 30*args.dim
    for seed in range(args.seed, args.seed+args.experiments):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"\nExperiment Seed: {seed})")
        best_values, gap_metrics, simple_regrets, cumulative_regrets = bayesian_optimization(n_iterations, seed, args.acquisition, args.kernel, args.function, args.dim)
        all_results.append([best_values, gap_metrics, simple_regrets, cumulative_regrets])
        print(f"Best value: {best_values[-1]:.4f}")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization on various test functions')
    # parser.add_argument('--iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--acquisition', type=str, default='LogEI', choices=['EI', 'LogEI', 'PI', 'LogPI', 'UCB', 'PM'], 
                        help='Acquisition function (EI: Expected Improvement, LogEI: Log Expected Improvement, PI: Probability of Improvement, LogPI, UCB: Upper Confidence Bound, PM: Posterior Mean)')
    parser.add_argument('--kernel', type=str, default='Matern52', choices=['Matern52', 'Matern32', 'RBF', 'RFF'], 
                        help='GP kernel (matern: Matérn kernel, rbf: Radial Basis Function kernel, rff: Random Fourier Features kernel)')
    parser.add_argument('--function', type=str, default='Hartmann', choices=list(true_maxima.keys()),
                        help='Test function to optimize')
    parser.add_argument('--dim', type=int, default=6, help='Dimension for functions that support variable dimensions')
    
    args = parser.parse_args()
    
    all_results = run_experiments(args)

    # Save results as .npy file
    all_results_np = np.array(all_results, dtype=object)
    np.save(f"baseline_function_{args.function}{args.dim}_kernel_{args.kernel}_acquisition_{args.acquisition}_optimization_results.npy", all_results_np)
    print(f"Results saved to baseline_function_{args.function}{args.dim}_kernel_{args.kernel}_acquisition_{args.acquisition}_optimization_results.npy")
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    LogExpectedImprovement
)
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.optim import optimize_acqf
from botorch_test_functions import setup_test_function, true_maxima
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RFFKernel
from botorch.acquisition.analytic import PosteriorMean
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

def get_next_points(train_X, train_Y, best_train_Y, bounds, acq_functions, kernel, n_points=1, gains=None, acq_weight='bandit'):
    base_kernel = {
        'Matern52': MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
        'RBF': RBFKernel(ard_num_dims=train_X.shape[-1]),
        'Matern32': MaternKernel(nu=1.5, ard_num_dims=train_X.shape[-1]),
        'RFF': RFFKernel(num_samples=1000, ard_num_dims=train_X.shape[-1])
    }[kernel]

    single_model = SingleTaskGP(train_X, train_Y, covar_module=ScaleKernel(base_kernel))
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    with gpytorch.settings.cholesky_jitter(1e-1):
        fit_gpytorch_mll(mll)

    acq_function_map = {
        'EI': ExpectedImprovement(model=single_model, best_f=best_train_Y),
        'UCB': UpperConfidenceBound(model=single_model, beta=0.1),
        'PI': ProbabilityOfImprovement(model=single_model, best_f=best_train_Y),
        'LogEI': LogExpectedImprovement(model=single_model, best_f=best_train_Y),
        'LogPI': LogProbabilityOfImprovement(model=single_model, best_f=best_train_Y),
        'PM': PosteriorMean(model=single_model)
    }

    candidates_list = []
    for acq_name in acq_functions:
        if acq_name in acq_function_map:
            acq_function = acq_function_map[acq_name]
            candidates, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=n_points,
                num_restarts=2,
                raw_samples=20,
                options={"batch_limit": 5, "maxiter": 200}
            )
            candidates_list.append(candidates)

    if not candidates_list:
        print("Warning: No valid acquisition functions. Using Expected Improvement.")
        ei = ExpectedImprovement(model=single_model, best_f=best_train_Y)
        candidates, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            q=n_points,
            num_restarts=2,
            raw_samples=20,
            options={"batch_limit": 5, "maxiter": 200}
        )
        candidates_list = [candidates]
        acq_functions = ['EI']

    if acq_weight == 'random' or gains is None or len(gains) == 0:
        chosen_acq_index = np.random.choice(len(candidates_list))
    else:  # bandit
        eta = 0.1
        logits = np.array(gains[:len(candidates_list)])
        logits -= np.max(logits)
        exp_logits = np.exp(eta * logits)
        probs = exp_logits / np.sum(exp_logits)
        chosen_acq_index = np.random.choice(len(candidates_list), p=probs)

    return candidates_list[chosen_acq_index], chosen_acq_index, single_model

def bayesian_optimization(args):
    num_iterations = 100
    initial_points = int(0.1 * num_iterations)
    objective, bounds = setup_test_function(args.function, dim=args.dim)
    bounds = bounds.to(dtype=dtype, device=device)

    # Draw initial points
    train_X = draw_sobol_samples(bounds=bounds, n=initial_points, q=1).squeeze(1)
    train_Y = objective(train_X).unsqueeze(-1)

    best_init_y = train_Y.max().item()
    best_train_Y = best_init_y

    true_max = true_maxima[args.function]

    gains = np.zeros(len(args.acquisition))
    max_values = [best_train_Y]
    gap_metrics = [gap_metric(best_init_y, best_init_y, true_max)]
    simple_regrets = [true_max - best_train_Y]
    cumulative_regrets = [true_max - best_train_Y]
    chosen_acq_functions = []
    kernel_names = []

    for i in range(num_iterations):
        print(f"Running iteration {i+1}/{num_iterations}, Best value = {best_train_Y:.4f}")

        # Compute bounds for normalization
        fit_bounds = torch.stack([torch.min(train_X, 0)[0], torch.max(train_X, 0)[0]])

        # Normalize inputs and standardize outputs
        train_X_normalized = normalize(train_X, bounds=fit_bounds)
        train_Y_standardized = standardize(train_Y)

        # Standardize best observed value
        best_f = (best_train_Y - train_Y.mean()) / train_Y.std()

        new_candidates_normalized, chosen_acq_index, model = get_next_points(
            train_X_normalized, train_Y_standardized, 
            best_f, normalize(bounds, fit_bounds),
            args.acquisition, args.kernel, 1, gains, args.acq_weight
        )

        # Unnormalize the candidates
        new_candidates = unnormalize(new_candidates_normalized, bounds=fit_bounds)
        new_Y = objective(new_candidates).unsqueeze(-1)

        train_X = torch.cat([train_X, new_candidates])
        train_Y = torch.cat([train_Y, new_Y])

        best_train_Y = train_Y.max().item()

        max_values.append(best_train_Y)
        gap_metrics.append(gap_metric(best_init_y, best_train_Y, true_max))
        simple_regrets.append(true_max - best_train_Y)
        cumulative_regrets.append(cumulative_regrets[-1] + (true_max - best_train_Y))
        chosen_acq_functions.append(args.acquisition[chosen_acq_index])
        kernel_names.append(args.kernel)

        posterior_mean = model.posterior(new_candidates_normalized).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return max_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_acq_functions, kernel_names

def run_experiments(args):
    all_results = []

    for seed in range(args.seed, args.seed+args.experiments):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        start_time = time.time()
        max_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_acq_functions, kernel_names = bayesian_optimization(args)
        end_time = time.time()

        experiment_time = end_time - start_time
        all_results.append([max_values, gap_metrics, simple_regrets, cumulative_regrets, experiment_time, chosen_acq_functions, kernel_names])

        print(f"Experiment {seed} for portfolio completed in {experiment_time:.2f} seconds")
        print(f"Acquisition functions used: {chosen_acq_functions[:5]}... (showing first 5)")
        print(f"Kernels used: {kernel_names[:5]}... (showing first 5)")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--acquisition', nargs='+', default=['LogEI', 'LogPI', 'UCB'], 
                        choices=['EI', 'UCB', 'PI', 'LogEI', 'PM', 'LogPI'],
                        help='List of acquisition functions to use')
    parser.add_argument('--kernel', type=str, default='Matern52', 
                        choices=['Matern52', 'RBF', 'Matern32', 'RFF'],
                        help='GP kernel to use')
    parser.add_argument('--experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--function', type=str, default='Hartmann', choices=list(true_maxima.keys()),
                        help='Test function to optimize')
    parser.add_argument('--dim', type=int, default=6, help='Dimensionality of the problem (for functions that support variable dimensions)')
    parser.add_argument('--acq_weight', type=str, default='bandit', choices=['random', 'bandit'],
                        help='Method for selecting acquisition function: random or bandit')
    args = parser.parse_args()
    acquisition_str = "_".join(args.acquisition)
    all_results = run_experiments(args)

    # Convert to numpy array and save
    all_results_np = np.array(all_results, dtype=object)
    os.makedirs(f"./Results/{args.function}", exist_ok=True)
    np.save(f"./Results/{args.function}/GPHedge_{args.acq_weight}.npy", all_results_np)

    print(f"Results saved to GPHedge_{args.acq_weight}.npy")
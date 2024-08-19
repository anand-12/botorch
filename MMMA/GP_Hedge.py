import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
# from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch_test_functions import setup_test_function, true_maxima
from botorch.utils.sampling import draw_sobol_samples

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def get_next_points(objective, train_X, train_Y, best_train_Y, bounds, acq_functions, n_points=1, gains=None):
    single_model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)

    acq_function_map = {
        'EI': ExpectedImprovement(model=single_model, best_f=best_train_Y),
        'UCB': UpperConfidenceBound(model=single_model, beta=0.1),
        'PI': ProbabilityOfImprovement(model=single_model, best_f=best_train_Y)
    }

    candidates_list = []
    for acq_name in acq_functions:
        if acq_name in acq_function_map:
            acq_function = acq_function_map[acq_name]
            candidates, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=n_points,
                num_restarts=20,
                raw_samples=512,
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
            num_restarts=20,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200}
        )
        candidates_list = [candidates]
        acq_functions = ['EI']

    if gains is None or len(gains) == 0:
        chosen_acq_index = np.random.choice(len(candidates_list))
    else:
        eta = 0.1
        logits = np.array(gains[:len(candidates_list)])
        logits -= np.max(logits)
        exp_logits = np.exp(eta * logits)
        probs = exp_logits / np.sum(exp_logits)
        chosen_acq_index = np.random.choice(len(candidates_list), p=probs)

    return candidates_list[chosen_acq_index], chosen_acq_index, single_model

def run_experiment(args):
    objective, bounds = setup_test_function(args.function, dim=args.dim)
    bounds = bounds.to(dtype=dtype, device=device)
    
    train_X = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
    train_Y = -objective(train_X).unsqueeze(-1)
    best_train_Y = train_Y.max().item()
    
    gains = np.zeros(len(args.acquisition))
    best_observed_values = [best_train_Y]
    chosen_acq_functions = []

    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}")
        new_candidates, chosen_acq_index, single_model = get_next_points(
            objective, train_X, train_Y, best_train_Y, bounds, args.acquisition, 1, gains
        )
        new_Y = -objective(new_candidates).unsqueeze(-1)

        train_X = torch.cat([train_X, new_candidates])
        train_Y = torch.cat([train_Y, new_Y])
        best_train_Y = train_Y.max().item()
        best_observed_values.append(best_train_Y)
        chosen_acq_functions.append(chosen_acq_index)

        posterior_mean = single_model.posterior(new_candidates).mean
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return best_observed_values, chosen_acq_functions

def plot_results(all_best_observed, all_chosen_acq, args):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, best_observed in enumerate(all_best_observed):
        plt.plot(best_observed, alpha=0.3)
    median_best_observed = np.median(all_best_observed, axis=0)
    plt.plot(median_best_observed, color='red', linewidth=2, label='Median')
    plt.axhline(y=true_maxima[args.function], color='green', linestyle='--', label='True Maximum')
    plt.title(f"Best Objective Function Value for {args.function}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Function Value")
    plt.legend()

    plt.subplot(1, 2, 2)
    acq_counts = np.zeros((args.iterations, len(args.acquisition)))
    for chosen_acq in all_chosen_acq:
        for i, acq in enumerate(chosen_acq):
            acq_counts[i, acq] += 1
    acq_percentages = acq_counts / args.experiments * 100

    for i, acq in enumerate(args.acquisition):
        plt.plot(acq_percentages[:, i], label=acq)
    plt.title("Acquisition Function Usage")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage Used")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BoTorch Bayesian Optimization')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--acquisition', nargs='+', default=['EI', 'UCB', 'PI'], 
                        choices=['EI', 'UCB', 'PI'],
                        help='List of acquisition functions to use (EI: Expected Improvement, UCB: Upper Confidence Bound, PI: Probability of Improvement)')
    parser.add_argument('--experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--function', type=str, default='Hartmann', choices=list(true_maxima.keys()),
                        help='Test function to optimize')
    parser.add_argument('--dim', type=int, default=6, help='Dimensionality of the problem (for functions that support variable dimensions)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_best_observed = []
    all_chosen_acq = []

    for i in range(args.experiments):
        print(f"Running experiment {i+1}/{args.experiments}")
        best_observed, chosen_acq = run_experiment(args)
        all_best_observed.append(best_observed)
        all_chosen_acq.append(chosen_acq)

    plot_results(all_best_observed, all_chosen_acq, args)
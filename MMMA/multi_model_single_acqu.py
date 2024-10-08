import argparse
import torch
import gpytorch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch_test_functions import setup_test_function, true_maxima
from botorch.acquisition.analytic import PosteriorMean, LogProbabilityOfImprovement
from botorch.utils.transforms import normalize, unnormalize, standardize
import warnings, os, random, time

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def fit_model(train_x, train_y, kernel):
    covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=train_x.shape[-1])).to(device)
    model = SingleTaskGP(train_x, train_y, covar_module=covar_module).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.cholesky_jitter(1e-1):  
        fit_gpytorch_mll(mll)
    return model, mll

def calculate_weights(models, mlls, train_x, train_y):
    log_likelihoods = []
    for model, mll in zip(models, mlls):
        if model is not None and mll is not None:
            with gpytorch.settings.cholesky_jitter(1e-1):
                ll = mll(model(train_x), train_y).sum().item()
                log_likelihoods.append(ll)
        else:
            log_likelihoods.append(float('-inf'))
    log_likelihoods = np.array(log_likelihoods)
    max_log_likelihood = np.max(log_likelihoods)
    log_likelihoods -= max_log_likelihood
    weights = np.exp(log_likelihoods)
    weights /= np.sum(weights)
    return weights

def select_model(weights):
    return np.random.choice(len(weights), p=weights)

def gap_metric(f_start, f_current, f_star):
    return np.abs((f_start - f_current) / (f_start - f_star))

def bayesian_optimization(args):
    n_iterations = 100
    initial_points = int(0.1 * n_iterations)
    function, bounds = setup_test_function(args.function, args.dim)
    bounds = bounds.to(dtype=dtype, device=device)
    true_max = true_maxima[args.function]

    train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=1).squeeze(1)
    train_y = function(train_x).unsqueeze(-1)
    best_init_y = train_y.max().item()
    best_observed_value = best_init_y

    max_values = [best_observed_value]
    gap_metrics = [gap_metric(best_init_y, best_init_y, true_max)]
    simple_regrets = [true_max - best_observed_value]
    cumulative_regrets = [true_max - best_observed_value]
    chosen_kernels = []
    chosen_acquisitions = []

    kernel_map = {
        'RBF': gpytorch.kernels.RBFKernel,
        'Matern52': lambda ard_num_dims: gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims),
        'Matern32': lambda ard_num_dims: gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims),
        'RFF': lambda ard_num_dims: gpytorch.kernels.RFFKernel(num_samples=1000, ard_num_dims=ard_num_dims)
    }

    for i in range(n_iterations):
        print(f"Running iteration {i+1}/{n_iterations}, Best value = {best_observed_value:.4f}")

        # Compute bounds for normalization
        fit_bounds = torch.stack([torch.min(train_x, 0)[0], torch.max(train_x, 0)[0]])

        # Normalize inputs and standardize outputs
        train_x_normalized = normalize(train_x, bounds=fit_bounds)
        train_y_standardized = standardize(train_y)

        models = []
        mlls = []
        for kernel in args.kernels:
            model, mll = fit_model(train_x_normalized, train_y_standardized, kernel_map[kernel])
            models.append(model)
            mlls.append(mll)


        if args.weight_type == 'uniform':
            selected_model_index = np.random.choice(len(models))
        elif args.weight_type == 'likelihood':
            weights = calculate_weights(models, mlls, train_x_normalized, train_y_standardized)
            selected_model_index = select_model(weights)
        else:
            raise ValueError(f"Unknown weight type: {args.weight_type}")
        model = models[selected_model_index]
        chosen_kernel = args.kernels[selected_model_index]

        # Standardize best observed value
        best_f = (best_observed_value - train_y.mean()) / train_y.std()

        acq_function_map = {
            'EI': ExpectedImprovement(model=model, best_f=best_f),
            'LogEI': LogExpectedImprovement(model=model, best_f=best_f),
            'PI': ProbabilityOfImprovement(model=model, best_f=best_f),
            'LogPI': LogProbabilityOfImprovement(model=model, best_f=best_f),
            'UCB': UpperConfidenceBound(model=model, beta=0.1),
            'PM': PosteriorMean(model=model)
        }
        acq_function = acq_function_map[args.acquisition]

        new_x_normalized, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=normalize(bounds, fit_bounds),
            q=1,
            num_restarts=2,
            raw_samples=20,
        )

        # Unnormalize new_x before evaluating objective
        new_x = unnormalize(new_x_normalized, bounds=fit_bounds)
        new_y = function(new_x).unsqueeze(-1)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_observed_value = train_y.max().item()

        max_values.append(best_observed_value)
        gap_metrics.append(gap_metric(best_init_y, best_observed_value, true_max))
        simple_regrets.append(true_max - best_observed_value)
        cumulative_regrets.append(cumulative_regrets[-1] + (true_max - best_observed_value))
        chosen_kernels.append(chosen_kernel)
        chosen_acquisitions.append(args.acquisition)

    return max_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_kernels, chosen_acquisitions

def run_experiments(args):
    all_results = []
    for seed in range(args.seed, args.seed + args.experiments):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        print(f"\nExperiment with seed {seed}")
        start = time.time()
        max_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_kernels, chosen_acquisitions = bayesian_optimization(args)
        end = time.time()
        experiment_time = end - start
        all_results.append([max_values, gap_metrics, simple_regrets, cumulative_regrets, experiment_time, chosen_kernels, chosen_acquisitions])
        print(f"Experiment time for multimodel single acquisition: {experiment_time:.2f} seconds")
        print(f"Kernels used: {chosen_kernels[:5]}... (showing first 5)")
        print(f"Acquisition functions used: {chosen_acquisitions[:5]}... (showing first 5)")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--kernels", nargs="+", default=["RBF"], 
                        choices=["RBF", "Matern52", "Matern32", "RFF"], 
                        help="List of kernels to use")
    parser.add_argument("--experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--function", type=str, default="Hartmann", choices=list(true_maxima.keys()), help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=6, help="Dimensionality of the test function")
    parser.add_argument("--acquisition", type=str, default="EI", choices=["EI", "PI", "LogPI", "UCB", "LogEI", "PM"], help="Acquisition function to use")
    parser.add_argument("--weight_type", type=str, default="uniform", choices=["uniform", "likelihood"], 
                        help="Type of weights to use for model selection or ensemble")

    args = parser.parse_args()

    if args.weight_type not in ["uniform", "likelihood"]:
        parser.error("--weight_type must be specified as either 'uniform' or 'likelihood'")

    all_results = run_experiments(args)

    kernel_str = "_".join(args.kernels)
    all_results_np = np.array(all_results, dtype=object)
    os.makedirs(f"./{args.function}", exist_ok=True)
    np.save(f"./{args.function}/MultiModel_{args.weight_type}.npy", np.array(all_results, dtype=object))
    print(f"\nResults saved to MultiModel_{args.weight_type}.npy")
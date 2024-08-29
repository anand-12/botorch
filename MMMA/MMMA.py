import argparse
import botorch
import gpytorch
import torch
import numpy as np
import os
import time
import warnings
import random
from botorch_test_functions import setup_test_function, true_maxima
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.models import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, PosteriorMean, LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.models.ensemble import EnsembleModel
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.transforms import normalize, unnormalize, standardize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
botorch.settings.debug = True
warnings.filterwarnings("ignore")

class MyEnsembleModel(EnsembleModel):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = models
        self._num_outputs = models[0].num_outputs
        self.weights = weights

    def forward(self, X):
        for model in self.models:
            model.eval()
        outputs = [model(X) for model in self.models]
        samples = torch.stack([output.rsample() for output in outputs], dim=0)
        return samples

    def posterior(self, X, output_indices=None, posterior_transform=None, **kwargs):
        values = self.forward(X)
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        if output_indices is not None:
            values = values[..., output_indices]
        posterior = EnsemblePosterior(values=values, my_weights=self.weights)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    @property
    def num_outputs(self):
        return self._num_outputs

def fit_model(train_x, train_y, kernel_type):
    train_x = train_x.to(dtype=torch.float64, device=device)
    train_y = train_y.to(dtype=torch.float64, device=device)
    ard_num_dims = train_x.shape[-1]

    kernel_map = {
        'RBF': gpytorch.kernels.RBFKernel,
        'Matern52': lambda ard_num_dims: gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims),
        'Matern32': lambda ard_num_dims: gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims),
        'RFF': lambda ard_num_dims: gpytorch.kernels.RFFKernel(num_samples=1024, num_dims=ard_num_dims)
    }

    covar_module = gpytorch.kernels.ScaleKernel(kernel_map[kernel_type](ard_num_dims=ard_num_dims)).to(device=device, dtype=torch.float64)

    class CustomGP(SingleTaskGP):
        def __init__(self, train_x, train_y):
            super().__init__(train_x, train_y)
            self.mean_module = gpytorch.means.ConstantMean().to(device=device, dtype=torch.float64)
            self.covar_module = covar_module

    model = CustomGP(train_x, train_y).to(device=device, dtype=torch.float64)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device=device, dtype=torch.float64)

    with gpytorch.settings.cholesky_jitter(1e-1):  
        fit_gpytorch_mll(mll)

    return model, mll

def calculate_weights(models, mlls, train_x, train_y):
    log_likelihoods = []
    for model, mll in zip(models, mlls):
        if model is not None and mll is not None:         
            with gpytorch.settings.cholesky_jitter(1e-2):
                ll = mll(model(train_x), train_y).sum().item()
                log_likelihoods.append(ll)    
        else:
            log_likelihoods.append(float('-inf'))

    log_likelihoods = np.array(log_likelihoods, dtype=np.float64)
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
    bounds = bounds.to(dtype=torch.float64, device=device)
    true_max = true_maxima[args.function]

    train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=1).squeeze(1)
    train_y = function(train_x).unsqueeze(-1)
    best_init_y = train_y.max().item()
    best_observed_value = best_init_y

    gains = np.zeros(len(args.acquisition), dtype=np.float64)
    eta = 0.1  

    best_observed_values = [best_observed_value]
    gap_metrics = [gap_metric(best_init_y, best_init_y, true_max)]
    simple_regrets = [true_max - best_observed_value]
    cumulative_regret = true_max - best_observed_value
    cumulative_regrets = [cumulative_regret]

    chosen_acq_functions = []
    selected_models = []

    for t in range(n_iterations):
        print(f"Running iteration {t+1}/{n_iterations}, Best value = {best_observed_value:.4f}")

        # Compute bounds for normalization
        fit_bounds = torch.stack([torch.min(train_x, 0)[0], torch.max(train_x, 0)[0]])

        # Normalize inputs and standardize outputs
        train_x_normalized = normalize(train_x, bounds=fit_bounds)
        train_y_standardized = standardize(train_y)

        models = [fit_model(train_x_normalized, train_y_standardized, kernel)[0] for kernel in args.kernels]
        mlls = [ExactMarginalLogLikelihood(model.likelihood, model).to(device=device, dtype=torch.float64) for model in models]

        if args.true_ensemble:
            if args.kernel_weight_type == 'uniform':
                weights = None
            elif args.kernel_weight_type == 'likelihood':
                weights = torch.tensor(calculate_weights(models, mlls, train_x_normalized, train_y_standardized), dtype=torch.float64, device=device)
            else:
                raise ValueError(f"Unknown weight type: {args.kernel_weight_type}")
            model = MyEnsembleModel(models, weights)
            selected_model = 'Ensemble'
        else:
            if args.kernel_weight_type == 'uniform':
                selected_model_index = np.random.choice(len(models))
            elif args.kernel_weight_type == 'likelihood':
                weights = calculate_weights(models, mlls, train_x_normalized, train_y_standardized)
                selected_model_index = select_model(weights)
            else:
                raise ValueError(f"Unknown weight type: {args.kernel_weight_type}")
            model = models[selected_model_index]
            selected_model = args.kernels[selected_model_index]

        # Standardize best observed value
        best_f = (best_observed_value - train_y.mean()) / train_y.std()

        acquisition = {
            'LogEI': LogExpectedImprovement(model=model, best_f=best_f),
            'EI': ExpectedImprovement(model=model, best_f=best_f),
            'LogPI': LogProbabilityOfImprovement(model=model, best_f=best_f),
            'UCB': UpperConfidenceBound(model=model, beta=0.1),
            'PI': ProbabilityOfImprovement(model=model, best_f=best_f)
        }

        candidates_list = []
        for acq_name in args.acquisition:
            acq_function = acquisition[acq_name]
            try:
                candidates, _ = optimize_acqf(
                    acq_function=acq_function, 
                    bounds=normalize(bounds, fit_bounds), 
                    q=1, 
                    num_restarts=2, 
                    raw_samples=20,
                    options={"batch_limit": 5, "maxiter": 200}
                )
            except RuntimeError as e:
                print(f"Optimization failed: {e}")
                discrete_candidates = torch.rand(1000, bounds.size(1), dtype=torch.float64, device=device) * (bounds[1] - bounds[0]) + bounds[0]
                discrete_candidates_normalized = normalize(discrete_candidates, bounds=fit_bounds)
                candidates, _ = optimize_acqf_discrete(
                    acq_function=acq_function,
                    q=1,
                    choices=discrete_candidates_normalized,
                )
            candidates_list.append(candidates)

        if args.acq_weight == 'random':
            chosen_acq_index = np.random.choice(len(args.acquisition))
        else:  # bandit
            logits = gains if len(gains) > 0 else np.ones(len(args.acquisition), dtype=np.float64)
            logits -= np.max(logits)
            exp_logits = np.exp(eta * logits)
            probs = exp_logits / np.sum(exp_logits)
            chosen_acq_index = np.random.choice(len(args.acquisition), p=probs)

        chosen_acq_functions.append(args.acquisition[chosen_acq_index])
        selected_models.append(selected_model)

        new_candidates_normalized = candidates_list[chosen_acq_index]
        new_candidates = unnormalize(new_candidates_normalized, bounds=fit_bounds)
        new_y = function(new_candidates).unsqueeze(-1)
        train_x = torch.cat([train_x, new_candidates])
        train_y = torch.cat([train_y, new_y])
        best_observed_value = train_y.max().item()

        best_observed_values.append(best_observed_value)
        gap_metrics.append(gap_metric(best_init_y, best_observed_value, true_max))
        simple_regrets.append(true_max - best_observed_value)
        cumulative_regrets.append(cumulative_regrets[-1] + (true_max - best_observed_value))

        if args.true_ensemble:
            posterior = model.posterior(new_candidates_normalized)
            posterior_mean = posterior.mean.mean(dim=0) 
        else:
            posterior_mean = model.posterior(new_candidates_normalized).mean

        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    return best_observed_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_acq_functions, selected_models

def run_experiments(args):
    all_results = []
    for seed in range(args.seed, args.seed + args.experiments):
        print(f"\nRunning experiment with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        start = time.time()
        best_observed_values, gap_metrics, simple_regrets, cumulative_regrets, chosen_acq_functions, selected_models = bayesian_optimization(args)
        end = time.time()
        experiment_time = end - start
        print(f"Experiment time for MMMA: {experiment_time:.2f} seconds")
        all_results.append([best_observed_values, gap_metrics, simple_regrets, cumulative_regrets, experiment_time, chosen_acq_functions, selected_models])
        print(f"Acquisition functions used: {chosen_acq_functions[:5]}... (showing first 5)")
        print(f"Models used: {selected_models[:5]}... (showing first 5)")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--acquisition", nargs="+", default=["LogEI"], choices=["LogEI", "EI", "UCB", "PI", "LogPI"], help="List of acquisition functions")
    parser.add_argument("--kernels", nargs="+", default=["Matern52"], choices=["Matern52", "RBF", "Matern32", "RFF"], help="List of kernels")
    parser.add_argument("--experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--function", type=str, default="Hartmann", choices=list(true_maxima.keys()), help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=6, help="Dimensionality of the test function")
    parser.add_argument("--true_ensemble", action="store_true", help="Use true ensemble model if set, otherwise use weighted model selection")
    parser.add_argument("--kernel_weight_type", type=str, default="uniform", choices=["uniform", "likelihood"], help="Type of weights to use for model selection or ensemble")
    parser.add_argument("--acq_weight", type=str, default="bandit", choices=["random", "bandit"], help="Method for selecting acquisition function: random or bandit")

    args = parser.parse_args()

    if args.kernel_weight_type not in ["uniform", "likelihood"]:
        parser.error("--kernel_weight_type must be specified as either 'uniform' or 'likelihood'")

    all_results = run_experiments(args)

    kernel_str = "_".join(args.kernels)
    acq_str = "_".join(args.acquisition)
    os.makedirs(f"./{args.function}", exist_ok=True)
    np.save(f"./{args.function}_2/MMMA_{args.kernel_weight_type}_{args.acq_weight}.npy", np.array(all_results, dtype=object))
    print(f"\nResults saved to MMMA_{args.kernel_weight_type}_{args.acq_weight}.npy")
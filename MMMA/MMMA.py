import argparse
import botorch
import gpytorch
import torch
import numpy as np
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

def gap_metric(f_start, f_current, f_star):
    return np.abs((f_start - f_current) / (f_start - f_star))

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

def get_next_points(train_x, train_y, best_observed_value, bounds, eta, n_points=1, gains=None, kernel_types=[], acq_func_types=[], true_ensemble=False, weight_type='uniform'):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    bounds = bounds.to(device)

    models = [fit_model(train_x, train_y, kernel)[0] for kernel in kernel_types]
    mlls = [ExactMarginalLogLikelihood(model.likelihood, model).to(device=device, dtype=torch.float64) for model in models]

    if weight_type == 'uniform':
        weights = None if true_ensemble else np.ones(len(models)) / len(models)
    elif weight_type == 'likelihood':
        weights = torch.tensor(calculate_weights(models, mlls, train_x, train_y), dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")

    if true_ensemble:
        model = MyEnsembleModel(models, weights)
    else:
        selected_model_index = select_model(weights)
        model = models[selected_model_index]

    acquisition = {
        'LogEI': LogExpectedImprovement(model=model, best_f=best_observed_value),
        'EI': ExpectedImprovement(model=model, best_f=best_observed_value),
        'LogPI': LogProbabilityOfImprovement(model=model, best_f=best_observed_value),
        'UCB': UpperConfidenceBound(model=model, beta=0.1),
        'PI': ProbabilityOfImprovement(model=model, best_f=best_observed_value)
    }
    acquisition_functions = [acquisition[acq] for acq in acq_func_types]
    
    candidates_list = []
    for acq_function in acquisition_functions:
        try:
            candidates, _ = optimize_acqf(
                acq_function=acq_function, 
                bounds=bounds, 
                q=n_points, 
                num_restarts=10, 
                raw_samples=32,
                options={"batch_limit": 5, "maxiter": 200}
            )
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            discrete_candidates = torch.rand(1000, bounds.size(1), dtype=torch.float64, device=device) * (bounds[1] - bounds[0]) + bounds[0]
            candidates, _ = optimize_acqf_discrete(
                acq_function=acq_function,
                q=n_points,
                choices=discrete_candidates,
            )
        candidates_list.append(candidates)

    logits = np.array(gains, dtype=np.float64) if gains is not None else np.ones(len(acquisition_functions), dtype=np.float64)
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    chosen_acq_index = np.random.choice(len(acquisition_functions), p=probs)

    return candidates_list[chosen_acq_index], chosen_acq_index, selected_model_index if not true_ensemble else None, model

def run_experiment(n_iterations, kernel_types, acq_func_types, function, bounds, true_max, true_ensemble, weight_type):
    start = time.time()
    bounds = bounds.to(dtype=torch.float64, device=device)

    # Generate initial data
    initial_points = int(0.1 * n_iterations)
    train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=1).squeeze(1)
    train_y = function(train_x).unsqueeze(-1)
    best_observed_value = train_y.max().item()
    
    gains = np.zeros(len(acq_func_types), dtype=np.float64)
    eta = 0.1  

    best_observed_values = [best_observed_value]
    chosen_acq_functions = []
    selected_models = []
    gap_metrics = [gap_metric(best_observed_value, best_observed_value, true_max)]
    simple_regrets = [true_max - best_observed_value]
    cumulative_regret = true_max - best_observed_value
    cumulative_regrets = [cumulative_regret]

    for t in range(n_iterations):
        new_candidates, chosen_acq_index, selected_model_index, model = get_next_points(
            train_x, train_y, best_observed_value, bounds, eta, 1, gains, kernel_types, acq_func_types, true_ensemble, weight_type)
             
        new_y = function(new_candidates).unsqueeze(-1)
        train_x = torch.cat([train_x, new_candidates])
        train_y = torch.cat([train_y, new_y])
        
        current_best_y = new_y.max().item()
        best_observed_value = max(best_observed_value, current_best_y)
        best_observed_values.append(best_observed_value)
        chosen_acq_functions.append(chosen_acq_index)
        selected_models.append(selected_model_index)
        
        g_i = gap_metric(best_observed_values[0], best_observed_value, true_max)
        gap_metrics.append(g_i)
        
        simple_regret = true_max - best_observed_value
        simple_regrets.append(simple_regret)
        
        cumulative_regret += (true_max - current_best_y)
        cumulative_regrets.append(cumulative_regret)
        
        print(f"Iteration {t+1}/{n_iterations}: Best value = {best_observed_value:.4f}, Simple Regret = {simple_regret:.4f}, Cumulative Regret = {cumulative_regret:.4f}")
        
        if true_ensemble:
            posterior = model.posterior(new_candidates)
            posterior_mean = posterior.mean.mean(dim=0)  # Average over ensemble members
        else:
            posterior_mean = model.posterior(new_candidates).mean

        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward

    execution_time = time.time() - start
    return best_observed_values, gap_metrics, simple_regrets, cumulative_regrets

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_iterations = 30 * args.dim
    function, bounds = setup_test_function(args.function, args.dim)
    true_max = true_maxima[args.function]

    all_results = []
    for seed in range(args.seed, args.seed + args.experiments):
        print(f"\nRunning experiment with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        s = time.time()
        experiment_results = run_experiment(num_iterations, args.kernels, args.acquisition, function, bounds, true_max, args.true_ensemble, args.weight_type)
        all_results.append(experiment_results)
        print(f"Time taken: {time.time() - s:.2f} seconds")
        print(f"Final Best value: {experiment_results[0][-1]:.4f}")

    kernel_str = "_".join(args.kernels)
    acq_str = "_".join(args.acquisition)
    # Save results
    np.save(f"{'True' if args.true_ensemble else 'False'}_{args.weight_type}_MMMA_function_{args.function}{args.dim}_kernel_{kernel_str}_acquisition_{acq_str}_optimization_results.npy", np.array(all_results, dtype=object))
    print(f"\nResults saved to {'True' if args.true_ensemble else 'False'}_{args.weight_type}_MMMA_function_{args.function}{args.dim}_kernel_{kernel_str}_acquisition_{acq_str}_optimization_results.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--acquisition", nargs="+", default=["LogEI"], choices=["LogEI", "EI", "UCB", "PI", "LogPI"], help="List of acquisition functions")
    parser.add_argument("--kernels", nargs="+", default=["Matern52"], choices=["Matern52", "RBF", "Matern32", "RFF"], help="List of kernels")
    parser.add_argument("--experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--function", type=str, default="Hartmann", choices=list(true_maxima.keys()), help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=6, help="Dimensionality of the test function")
    parser.add_argument("--true_ensemble", action="store_true", help="Use true ensemble model if set, otherwise use weighted model selection")
    parser.add_argument("--weight_type", type=str, default="uniform", choices=["uniform", "likelihood"], help="Type of weights to use for model selection or ensemble")

    args = parser.parse_args()

    if args.weight_type not in ["uniform", "likelihood"]:
        parser.error("--weight_type must be specified as either 'uniform' or 'likelihood'")

    main(args)
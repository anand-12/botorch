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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
botorch.settings.debug = True
warnings.filterwarnings("ignore")

from botorch.models.ensemble import EnsembleModel
from botorch.posteriors.ensemble import EnsemblePosterior

def target_function(test_func):
    def wrapper(individuals):
        result = []
        for x in individuals:
            result.append(-1.0 * test_func(x))
        return torch.tensor(result, dtype=torch.float64, device=device)
    return wrapper

def generate_initial_data(n, n_dim, test_func):
    train_x = draw_sobol_samples(bounds=torch.tensor([[0.0] * n_dim, [1.0] * n_dim], device=device), n=n, q=1).squeeze(1)
    exact_obj = torch.tensor([test_func(x.unsqueeze(0)) for x in train_x], dtype=torch.float64, device=device).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return train_x, exact_obj, best_observed_value

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
        # means = torch.stack([output.mean for output in outputs], dim=0)
        samples = torch.stack([output.rsample() for output in outputs], dim=0)
        return samples

    def posterior(self, X, output_indices=None, posterior_transform=None, **kwargs):
        values = self.forward(X)
        # Ensure values is 3D: [num_models, num_samples, num_outputs]
        if values.dim() == 2:
            # print("anand cabron")
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

    if kernel_type == 'RBF':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)).to(device=device, dtype=torch.float64)
    elif kernel_type == 'Matern52':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)).to(device=device, dtype=torch.float64)
    elif kernel_type == 'Matern32':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)).to(device=device, dtype=torch.float64)
    elif kernel_type == 'RQ':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=ard_num_dims)).to(device=device, dtype=torch.float64)
    elif kernel_type == 'RFF':
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RFFKernel(num_samples=1024, num_dims=ard_num_dims)
        ).to(device=device, dtype=torch.float64)
    elif kernel_type == "ConstantKernel":
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel()).to(device=device, dtype=torch.float64)
    elif kernel_type == "PeriodicKernel":
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(ard_num_dims=ard_num_dims)).to(device=device, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

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
    for i, (model, mll) in enumerate(zip(models, mlls)):
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

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=[], acq_func_types=[], true_ensemble=False, weight_type='uniform'):
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

    with gpytorch.settings.cholesky_jitter(1e-2):
        LogEI = LogExpectedImprovement(model=model, best_f=best_init_y)
        EI = ExpectedImprovement(model=model, best_f=best_init_y)
        UCB = UpperConfidenceBound(model=model, beta=0.1)
        PI = ProbabilityOfImprovement(model=model, best_f=best_init_y)
        LogPI = LogProbabilityOfImprovement(model=model, best_f=best_init_y)
        PM = PosteriorMean(model=model)

    acquisition = {'LogEI': LogEI, 'EI': EI, 'qEI': qEI, 'UCB': UCB, 'PI': PI, 'PM': PM}
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


def update_data(train_x, train_y, new_x, new_y):
    train_x = torch.cat([train_x, new_x]).to(dtype=torch.float64)
    train_y = torch.cat([train_y, new_y]).to(dtype=torch.float64)
    return train_x, train_y

def run_experiment(n_iterations, kernel_types, acq_func_types, initial_data, test_func, true_ensemble, weight_type):
    start = time.time()
    bounds = initial_data["bounds"].to(dtype=torch.float64, device=device)
    train_x, train_y = initial_data["train_x"].to(dtype=torch.float64, device=device), initial_data["train_y"].to(dtype=torch.float64, device=device)
    best_f = initial_data["best_init_y"]
    true_maximum = initial_data["true_maximum"]
    
    gains = np.zeros(len(acq_func_types), dtype=np.float64)
    eta = 0.1  

    best_observed_values = []
    chosen_acq_functions = []
    selected_models = []
    gap_metrics = []
    simple_regrets = []
    cumulative_regret = 0.0
    cumulative_regrets = []

    for t in range(n_iterations):
        new_candidates, chosen_acq_index, selected_model_index, model = get_next_points(
            train_x, train_y, best_f, bounds, eta, 1, gains, kernel_types, acq_func_types, true_ensemble, weight_type)
             
        new_results = test_func(new_candidates).unsqueeze(-1).to(dtype=torch.float64)
        train_x, train_y = update_data(train_x, train_y, new_candidates, new_results)
        
        current_best_f = train_y.max().item()
        best_f = max(best_f, current_best_f)  
        best_observed_values.append(best_f)
        chosen_acq_functions.append(chosen_acq_index)
        selected_models.append(selected_model_index)
        
        g_i = gap_metric(initial_data["best_init_y"], best_f, true_maximum)
        gap_metrics.append(g_i)
        
        simple_regret = true_maximum - best_f
        simple_regrets.append(simple_regret)
        
        cumulative_regret += (true_maximum - current_best_f)
        cumulative_regrets.append(cumulative_regret)
        
        print(f"Iteration {t+1}/{n_iterations}: Best value = {best_f:.4f}, Simple Regret = {simple_regret:.4f}, Cumulative Regret = {cumulative_regret:.4f}")
        
        if true_ensemble:
            posterior = model.posterior(new_candidates)
            posterior_mean = posterior.mean.mean(dim=0)  # Average over ensemble members
        else:
            posterior_mean = model.posterior(new_candidates).mean

        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward
    execution_time = time.time() - start
    # return (best_observed_values, chosen_acq_functions, selected_models, true_maximum, 
    #         gap_metrics, execution_time, simple_regrets, cumulative_regrets)
    # print(f"Type of best_observed_values: {type(best_observed_values)} type of gap_metrics: {type(gap_metrics)} type of simple_regrets: {type(simple_regrets)} type of cumulative_regrets: {type(cumulative_regrets)}")
    return best_observed_values, gap_metrics, simple_regrets, cumulative_regrets

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_iterations = 15 * args.dim
    initial_points = int(0.1 * num_iterations)
    test_function, bounds = setup_test_function(args.test_function, args.dim)
    true_max = true_maxima[args.test_function]
    target_func = target_function(test_function)

    all_results = []
    for seed in range(args.seed, args.seed + args.experiments):
        print(f"\nRunning experiment with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        init_x, init_y, best_init_y = generate_initial_data(initial_points, n_dim=bounds.size(1), test_func=target_func)
        initial_data = {
            "train_x": init_x.to(dtype=torch.float64, device=device),
            "train_y": init_y.to(dtype=torch.float64, device=device),
            "best_init_y": best_init_y,
            "bounds": bounds.to(dtype=torch.float64, device=device),
            "true_maximum": true_max
        }

        s = time.time()
        experiment_results = run_experiment(num_iterations, args.kernels, args.acquisition, initial_data, target_func, args.true_ensemble, args.weight_type)
        all_results.append(experiment_results)
        print(f"Time taken: {time.time() - s:.2f} seconds")
        print(f"Final Best value: {experiment_results[0][-1]:.4f}")

    # Save results
    np.save(f"{'True' if args.true_ensemble else 'False'}_{args.weight_type}_MMMA_{args.test_function}_optimization_results.npy", np.array(all_results, dtype=object))
    print(f"\nResults saved to {'true' if args.true_ensemble else 'false'}_{args.weight_type}_MMMA_{args.test_function}_optimization_results.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    # parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=125, help="Starting random seed")
    parser.add_argument("--acquisition", nargs="+", default=["LogEI"], choices=["LogEI", "EI", "UCB", "PI", "LogPI"], help="List of acquisition functions")
    parser.add_argument("--kernels", nargs="+", default=["Matern52"], choices=["Matern52", "RBF", "Matern32", "RFF"], help="List of kernels")
    parser.add_argument("--experiments", type=int, default=10, help="Number of experiments to run")
    parser.add_argument("--test_function", type=str, default="Hartmann", choices=list(true_maxima.keys()), help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=6, help="Dimensionality of the test function")
    parser.add_argument("--true_ensemble", action="store_true", help="Use true ensemble model if set, otherwise use weighted model selection")
    parser.add_argument("--weight_type", type=str, default="uniform", choices=["uniform", "likelihood"], help="Type of weights to use for model selection or ensemble")

    args = parser.parse_args()

    if args.weight_type not in ["uniform", "likelihood"]:
        parser.error("--weight_type must be specified as either 'uniform' or 'likelihood'")

    main(args)
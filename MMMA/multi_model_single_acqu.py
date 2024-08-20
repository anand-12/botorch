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
import warnings
from botorch.acquisition.analytic import PosteriorMean
from botorch.models.ensemble import EnsembleModel
from botorch.posteriors.ensemble import EnsemblePosterior

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

class MyEnsembleModel(EnsembleModel):
    def __init__(self, train_x, train_y, kernel_types):
        super().__init__()
        self.models = [self.fit_model(train_x, train_y, kernel) for kernel in kernel_types]
        self._num_outputs = train_y.shape[-1]

    def fit_model(self, train_x, train_y, kernel):
        covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=train_x.shape[-1])).to(device)
        model = SingleTaskGP(train_x, train_y, covar_module=covar_module).to(device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

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
        posterior = EnsemblePosterior(values=values)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    @property
    def num_outputs(self):
        return self._num_outputs

def calculate_weights(models, mlls, train_x, train_y):
    log_likelihoods = []
    for model, mll in zip(models, mlls):
        if model is not None and mll is not None:
            with gpytorch.settings.cholesky_jitter(1e-2):
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

def run_experiment(n_iterations, kernels, function, bounds, true_max, true_ensemble, acq_func_name):
    train_x = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
    train_y = -function(train_x).unsqueeze(-1)
    best_init_y = train_y.max().item()
    best_observed_value = best_init_y

    max_values = [best_observed_value]
    gap_metrics = [gap_metric(best_init_y, best_init_y, true_max)]
    simple_regrets = [true_max - best_observed_value]
    cumulative_regrets = [true_max - best_observed_value]

    kernel_map = {
        'RBF': gpytorch.kernels.RBFKernel,
        'Matern52': lambda ard_num_dims: gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims),
        'Matern32': lambda ard_num_dims: gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims),
        'RFF': lambda ard_num_dims: gpytorch.kernels.RFFKernel(num_samples=1000, ard_num_dims=ard_num_dims)
    }

    for i in range(n_iterations):
        if true_ensemble:
            model = MyEnsembleModel(train_x, train_y, [kernel_map[k] for k in kernels])
        else:
            models = []
            mlls = []
            for kernel in kernels:
                covar_module = gpytorch.kernels.ScaleKernel(
                    kernel_map[kernel](ard_num_dims=bounds.shape[-1])
                ).to(device)
                model = SingleTaskGP(train_x, train_y, covar_module=covar_module).to(device)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                models.append(model)
                mlls.append(mll)

            weights = calculate_weights(models, mlls, train_x, train_y)
            selected_model_index = select_model(weights)
            model = models[selected_model_index]

        acq_function_map = {
            'EI': ExpectedImprovement(model=model, best_f=best_observed_value),
            'LogEI': LogExpectedImprovement(model=model, best_f=best_observed_value),
            'PI': ProbabilityOfImprovement(model=model, best_f=best_observed_value),
            'UCB': UpperConfidenceBound(model=model, beta=0.1),
            'PM': PosteriorMean(model=model)
        }
        acq_function = acq_function_map[acq_func_name]

        new_x, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        new_y = -function(new_x).unsqueeze(-1)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_observed_value = train_y.max().item()
        
        max_values.append(best_observed_value)
        gap_metrics.append(gap_metric(best_init_y, best_observed_value, true_max))
        simple_regrets.append(true_max - best_observed_value)
        cumulative_regrets.append(cumulative_regrets[-1] + (true_max - new_y.item()))
        
        print(f"Iteration {i+1}: Best value = {best_observed_value:.4f}")
    print(f"Type of max_values is {type(max_values)}, type of gap_metrics is {type(gap_metrics)}, type of simple_regrets is {type(simple_regrets)}, type of cumulative_regrets is {type(cumulative_regrets)}")
    return max_values, gap_metrics, simple_regrets, cumulative_regrets

def main(args):
    function, bounds = setup_test_function(args.function, args.dim)
    bounds = bounds.to(dtype=dtype, device=device)
    true_max = true_maxima[args.function]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # print(args.true_ensemble)
    all_results = []
    for i in range(args.experiments):
        print(f"\nExperiment {i+1}/{args.experiments}")
        experiment_results = run_experiment(args.iterations, args.kernels, function, bounds, true_max, args.true_ensemble, args.acquisition)
        all_results.append(experiment_results)
        print(f"Final Best value: {experiment_results[0][-1]:.4f}")

    # Save results
    np.save(f"{args.true_ensemble}_ensemble_{args.function}_optimization_results.npy", np.array(all_results, dtype=object))
    print(f"\nResults saved to {args.true_ensemble}_ensemble_{args.function}_optimization_results.npy")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kernels", nargs="+", default=["RBF"], 
                        choices=["RBF", "Matern52", "Matern32", "RFF"], 
                        help="List of kernels to use")
    parser.add_argument("--experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--function", type=str, default="Hartmann", choices=list(true_maxima.keys()), help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=6, help="Dimensionality of the test function")
    parser.add_argument("--true_ensemble", action="store_true", help="Use true ensemble model if set, otherwise use weighted model selection")
    parser.add_argument("--acquisition", type=str, default="EI", choices=["EI", "PI", "UCB", "LogEI", "PM"], help="Acquisition function to use")

    args = parser.parse_args()
    main(args)
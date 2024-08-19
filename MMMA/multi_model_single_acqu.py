import argparse
import torch
import gpytorch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch_test_functions import setup_test_function, true_maxima
import warnings

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
        covar_module = gpytorch.kernels.ScaleKernel(
            getattr(gpytorch.kernels, kernel)(ard_num_dims=train_x.shape[-1])
        ).to(device)
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

def get_acquisition_function(acq_func_name, model, best_f):
    if acq_func_name == "EI":
        return ExpectedImprovement(model=model, best_f=best_f)
    elif acq_func_name == "PI":
        return ProbabilityOfImprovement(model=model, best_f=best_f)
    elif acq_func_name == "UCB":
        return UpperConfidenceBound(model=model, beta=0.1)
    else:
        raise ValueError(f"Unknown acquisition function: {acq_func_name}")

def run_experiment(n_iterations, kernels, test_function, bounds, true_max, true_ensemble, acq_func_name):
    train_x = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
    train_y = -test_function(train_x).unsqueeze(-1)
    best_observed_value = train_y.max().item()

    for i in range(n_iterations):
        if true_ensemble:
            model = MyEnsembleModel(train_x, train_y, kernels)
        else:
            models = []
            mlls = []
            for kernel in kernels:
                covar_module = gpytorch.kernels.ScaleKernel(
                    getattr(gpytorch.kernels, kernel)(ard_num_dims=bounds.shape[-1])
                ).to(device)
                model = SingleTaskGP(train_x, train_y, covar_module=covar_module).to(device)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                models.append(model)
                mlls.append(mll)

            weights = calculate_weights(models, mlls, train_x, train_y)
            selected_model_index = select_model(weights)
            model = models[selected_model_index]

        acq_function = get_acquisition_function(acq_func_name, model, best_observed_value)
        new_x, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        new_y = -test_function(new_x).unsqueeze(-1)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        best_observed_value = train_y.max().item()
        
        print(f"Iteration {i+1}: Best value = {best_observed_value:.4f}, True max = {true_max:.4f}")

    return best_observed_value

def main(args):
    test_function, bounds = setup_test_function(args.test_function, args.dim)
    bounds = bounds.to(dtype=dtype, device=device)
    true_max = true_maxima[args.test_function]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    for i in range(args.num_experiments):
        print(f"\nExperiment {i+1}/{args.num_experiments}")
        best_value = run_experiment(args.iterations, args.kernels, test_function, bounds, true_max, args.true_ensemble, args.acq_func)
        results.append(best_value)
        print(f"Final Best value: {best_value:.4f}")

    print("\nSummary of results:")
    print(f"Test function: {args.test_function}")
    print(f"True maximum: {true_max}")
    print(f"Mean best value: {np.mean(results):.4f}")
    print(f"Std dev of best values: {np.std(results):.4f}")
    print(f"Min best value: {np.min(results):.4f}")
    print(f"Max best value: {np.max(results):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization Experiment")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kernels", nargs="+", default=["MaternKernel"], help="List of kernels to use")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--test_function", type=str, default="Hartmann", choices=list(true_maxima.keys()), help="Test function to optimize")
    parser.add_argument("--dim", type=int, default=6, help="Dimensionality of the test function")
    parser.add_argument("--true_ensemble", action="store_true", help="Use true ensemble model if set, otherwise use weighted model selection")
    parser.add_argument("--acq_func", type=str, default="EI", choices=["EI", "PI", "UCB", "qEI"], help="Acquisition function to use")

    args = parser.parse_args()
    main(args)
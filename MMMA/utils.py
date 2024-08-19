import botorch
import gpytorch
import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement, LogExpectedImprovement, qExpectedImprovement, qAnalyticProbabilityOfImprovement, PosteriorMean
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import warnings
import time
from botorch.optim import optimize_acqf_discrete
from botorch.utils.sampling import draw_sobol_samples
botorch.settings.debug = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from botorch.models.ensemble import EnsembleModel
from botorch.posteriors.ensemble import EnsemblePosterior

class MyEnsembleModel(EnsembleModel):
    def __init__(self, train_x, train_y, kernel_types, my_weights = None):
        super().__init__()
        self.models = [fit_model(train_x, train_y, kernel)[0] for kernel in kernel_types]
        self._num_outputs = train_y.shape[-1]
        self.my_weights = my_weights

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
        posterior = EnsemblePosterior(values=values, my_weights=self.my_weights)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    @property
    def num_outputs(self):
        return self._num_outputs
    
def gap_metric(f_start, f_current, f_star):
    return np.abs((f_start - f_current) / (f_start - f_star))

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

def fit_model(train_x, train_y, kernel_type):
    train_x = train_x.to(dtype=torch.float64, device=device)
    train_y = train_y.to(dtype=torch.float64, device=device)

    if kernel_type == 'RBF':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(device=device, dtype=torch.float64)
    elif kernel_type == 'Matern52':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()).to(device=device, dtype=torch.float64)
    elif kernel_type == 'Matern32':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)).to(device=device, dtype=torch.float64)
    elif kernel_type == 'RQ':
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()).to(device=device, dtype=torch.float64)
    elif kernel_type == 'RFF':
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RFFKernel(num_samples=1024, num_dims=train_x.size(-1))
        ).to(device=device, dtype=torch.float64)
    elif kernel_type == "ConstantKernel":
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel()).to(device=device, dtype=torch.float64)
    elif kernel_type == "PeriodicKernel":
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()).to(device=device, dtype=torch.float64)
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
    
    log_likelihoods = np.array(log_likelihoods)
    max_log_likelihood = np.max(log_likelihoods)
    log_likelihoods -= max_log_likelihood  
    weights = np.exp(log_likelihoods)
    weights /= np.sum(weights)
    return weights

def select_model(weights):
    return np.random.choice(len(weights), p=weights)

def get_next_points(train_x, train_y, best_init_y, bounds, eta, n_points=1, gains=None, kernel_types=[], acq_func_types=[]):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    bounds = bounds.to(device)
    if len(kernel_types) == 1:
        weights = torch.tensor([1.0], dtype=torch.float64, device=device)
    elif len(kernel_types) == 2:
        weights = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
    elif len(kernel_types) == 3:
        weights = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    else:
        weights = torch.ones(len(kernel_types), dtype=torch.float64, device=device) / len(kernel_types)

    #FOR NOW
    ensemble_model = MyEnsembleModel(train_x, train_y, kernel_types, None)

    with gpytorch.settings.cholesky_jitter(1e-2):
        logEI = LogExpectedImprovement(model=ensemble_model, best_f=best_init_y)
        EI = ExpectedImprovement(model=ensemble_model, best_f=best_init_y)
        qEI = qExpectedImprovement(model=ensemble_model, best_f=best_init_y, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])))
        UCB = UpperConfidenceBound(model=ensemble_model, beta=0.1)
        PI = ProbabilityOfImprovement(model=ensemble_model, best_f=best_init_y)
        PM = PosteriorMean(model=ensemble_model)


    acq_funcs = {'logEI': logEI, 'EI': EI, 'qEI': qEI, 'UCB': UCB, 'PI': PI, 'PM': PM}
    acquisition_functions = [acq_funcs[acq] for acq in acq_func_types]
    
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

    logits = np.array(gains) if gains is not None else np.ones(len(acquisition_functions))
    logits -= np.max(logits)
    exp_logits = np.exp(eta * logits)
    probs = exp_logits / np.sum(exp_logits)
    chosen_acq_index = np.random.choice(len(acquisition_functions), p=probs)

    return candidates_list[chosen_acq_index], chosen_acq_index, None, ensemble_model


def update_data(train_x, train_y, new_x, new_y):
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])
    return train_x, train_y

def run_experiment(n_iterations, kernel_types, acq_func_types, initial_data, test_func):
    start = time.time()
    bounds = initial_data["bounds"].to(device)
    train_x, train_y = initial_data["train_x"].to(device), initial_data["train_y"].to(device)
    best_f = initial_data["best_init_y"]
    true_maximum = initial_data["true_maximum"]
    
    gains = np.zeros(len(acq_func_types))
    eta = 0.1  

    best_observed_values = []
    chosen_acq_functions = []
    selected_models = []
    gap_metrics = []
    simple_regrets = []
    cumulative_regret = 0
    cumulative_regrets = []

    for t in range(n_iterations):
        new_candidates, chosen_acq_index, _, ensemble_model = get_next_points(
            train_x, train_y, best_f, bounds, eta, 1, gains, kernel_types, acq_func_types)


        
        
        new_results = test_func(new_candidates).unsqueeze(-1)
        train_x = torch.cat([train_x, new_candidates])
        train_y = torch.cat([train_y, new_results])
        
        current_best_f = train_y.max().item()
        best_f = max(best_f, current_best_f)  
        best_observed_values.append(best_f)
        chosen_acq_functions.append(chosen_acq_index)
        # selected_models.append(selected_model_index)
        
        g_i = gap_metric(initial_data["best_init_y"], best_f, true_maximum)
        gap_metrics.append(g_i)
        
        
        simple_regret = true_maximum - best_f
        simple_regrets.append(simple_regret)
        
        
        cumulative_regret += (true_maximum - current_best_f)
        cumulative_regrets.append(cumulative_regret)
        
        print(f"Iteration {t+1}/{n_iterations}: Best value = {best_f:.4f}, Simple Regret = {simple_regret:.4f}, Cumulative Regret = {cumulative_regret:.4f}")
        
        posterior = ensemble_model.posterior(new_candidates)
        posterior_mean = posterior.mean.mean(dim=0)  # Average over ensemble members
        reward = posterior_mean.mean().item()
        gains[chosen_acq_index] += reward


    
    execution_time = time.time() - start
    return (best_observed_values, chosen_acq_functions, selected_models, true_maximum, 
            gap_metrics, execution_time, simple_regrets, cumulative_regrets)


import torch, numpy as np, random, time
from botorch.test_functions import (
    Ackley, 
    Beale,
    Branin, 
    Cosine8,
    DropWave,
    Griewank,
    Hartmann,
    Levy, 
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    # Schwefel,
    SixHumpCamel,
    StyblinskiTang,
    ThreeHumpCamel
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_test_function(func_name, dim=2):
    function_configs = {
        "Ackley": (Ackley(dim=dim), torch.tensor([[-32.768] * dim, [32.768] * dim])),
        "Beale": (Beale(), torch.tensor([[-4.5, -4.5], [4.5, 4.5]])),
        "Branin": (Branin(), torch.tensor([[-5, 0], [10, 15]])),
        "Cosine8": (Cosine8(), torch.tensor([[-1] * 8, [1] * 8])),
        "DropWave": (DropWave(), torch.tensor([[-5.12, -5.12], [5.12, 5.12]])),
        "Griewank": (Griewank(dim=dim), torch.tensor([[-600] * dim, [600] * dim])),
        "Hartmann": (Hartmann(dim=6), torch.tensor([[0] * 6, [1] * 6])),
        "Levy": (Levy(dim=dim), torch.tensor([[-10] * dim, [10] * dim])),
        "Michalewicz": (Michalewicz(dim=dim), torch.tensor([[0] * dim, [np.pi] * dim])),
        "Powell": (Powell(dim=dim), torch.tensor([[-4] * dim, [5] * dim])),
        "Rastrigin": (Rastrigin(dim=dim), torch.tensor([[-5.12] * dim, [5.12] * dim])),
        "Rosenbrock": (Rosenbrock(dim=dim), torch.tensor([[-5] * dim, [10] * dim])),
        # "Schwefel": (Schwefel(dim=dim), torch.tensor([[-500] * dim, [500] * dim])),
        "SixHumpCamel": (SixHumpCamel(), torch.tensor([[-3, -2], [3, 2]])),
        "StyblinskiTang": (StyblinskiTang(dim=dim), torch.tensor([[-5] * dim, [5] * dim])),
        "ThreeHumpCamel": (ThreeHumpCamel(), torch.tensor([[-5, -5], [5, 5]]))
    }

    if func_name not in function_configs:
        raise ValueError(f"Unsupported test function: {func_name}")

    func, bounds = function_configs[func_name]
    bounds = bounds.to(dtype=torch.float64, device=device)
    return func, bounds

true_maxima = {
    "Ackley": 0.0,
    "Beale": 0.0,
    "Branin": 0.397887,
    "Cosine8": 0.8,
    "DropWave": 0.0,
    "Griewank": 0.0,
    "Hartmann": 3.32237,
    "Levy": 0.0,
    "Michalewicz": 1.8013,  # for dim=2, changes with dimension
    "Powell": 0.0,
    "Rastrigin": 0.0,
    "Rosenbrock": 0.0,
    # "Schwefel": 0.0,
    "SixHumpCamel": -1.0316,
    "StyblinskiTang": 39.16599 * 2,  # for dim=2, scales with dimension
    "ThreeHumpCamel": 0.0
}


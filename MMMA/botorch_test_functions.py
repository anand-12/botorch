import torch, numpy as np, random, time
from botorch.test_functions import (
    Ackley, 
    Beale,
    Branin, 
    Bukin,
    Cosine8,
    DropWave,
    DixonPrice,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy, 
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SixHumpCamel,
    StyblinskiTang,
    ThreeHumpCamel
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_test_function(func_name, dim=2):
    function_configs = {
        "Ackley": (Ackley(dim=dim, negate=True), torch.tensor([[-32.768] * dim, [32.768] * dim])),
        "Beale": (Beale(negate=True), torch.tensor([[-4.5, -4.5], [4.5, 4.5]])),
        "Branin": (Branin(negate=True), torch.tensor([[-5, 0], [10, 15]])),
        "Bukin": (Bukin(negate=True), torch.tensor([[-15, -3], [-5, 3]])),
        "Cosine8": (Cosine8(), torch.tensor([[-1] * 8, [1] * 8])),
        "DropWave": (DropWave(negate=True), torch.tensor([[-5.12, -5.12], [5.12, 5.12]])),
        "DixonPrice": (DixonPrice(dim=dim, negate=True), torch.tensor([[-10] * dim, [10] * dim])),
        "EggHolder": (EggHolder(negate=True), torch.tensor([[-512, -512], [512, 512]])), 
        "Griewank": (Griewank(dim=dim, negate=True), torch.tensor([[-600] * dim, [600] * dim])),
        "Hartmann": (Hartmann(dim=6, negate=True), torch.tensor([[0] * 6, [1] * 6])),
        "HolderTable": (HolderTable(negate=True), torch.tensor([[-10, -10], [10, 10]])),
        "Levy": (Levy(dim=dim, negate=True), torch.tensor([[-10] * dim, [10] * dim])), #CHECKED TILL HERE, CHECK BELOW FOR NEGATE
        "Michalewicz": (Michalewicz(dim=dim, negate=True), torch.tensor([[0] * dim, [np.pi] * dim])),
        "Powell": (Powell(dim=dim, negate=True), torch.tensor([[-4] * dim, [5] * dim])),
        "Rastrigin": (Rastrigin(dim=dim, negate=True), torch.tensor([[-5.12] * dim, [5.12] * dim])),
        "Rosenbrock": (Rosenbrock(dim=dim, negate=True), torch.tensor([[-5] * dim, [10] * dim])),
        "Shekel": (Shekel(negate=True), torch.tensor([[0] * dim, [10] * dim])),
        "SixHumpCamel": (SixHumpCamel(negate=True), torch.tensor([[-3, -2], [3, 2]])),
        "StyblinskiTang": (StyblinskiTang(dim=dim), torch.tensor([[-5] * dim, [5] * dim])),
        "ThreeHumpCamel": (ThreeHumpCamel(negate=True), torch.tensor([[-5, -5], [5, 5]]))
    }

    if func_name not in function_configs:
        raise ValueError(f"Unsupported test function: {func_name}")

    func, bounds = function_configs[func_name]
    bounds = bounds.to(dtype=torch.float64, device=device)
    return func, bounds

true_maxima = {
    "Ackley": 0.0,
    "Beale": 0.0,
    "Branin": -0.397887,
    "Bukin": 0.0,
    "Cosine8": 0.8,
    "DropWave": 1.0,
    "DixonPrice": 0.0,
    "EggHolder": 959.6407,
    "Griewank": 0.0,
    "Hartmann": 3.32237,
    "HolderTable": 19.2085,
    "Levy": 0.0,
    "Michalewicz": 1.8013, 
    "Powell": 0.0,
    "Rastrigin": 0.0,
    "Rosenbrock": 0.0,
    "Shekel": 10.5363,
    "SixHumpCamel": 1.0316,
    "StyblinskiTang": 39.16599 * 2,
    "ThreeHumpCamel": 0.0
}


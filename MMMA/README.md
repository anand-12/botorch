baseline.py -> Implements standard bayesian optimization. 
multi_model_single_acqu.py -> Implements BO where the kernel is selected via BMA or random selection
GP_Hedge.py -> Implements the GP-Hedge algorithm. AF can either be selected via a bandit algorithm or random selection
MMMA.py -> Implements the MMMA algorithm
serial_execution.sh -> Shell script containing example executions

Additional details regarding how to specify the number of experiments, seeds, AF, kernel, etc can be found in the parser arguments of each file

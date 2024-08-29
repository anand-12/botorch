#!/bin/bash

# Function to run command and print output
run_and_print() {
    echo "Running $1"
    start_time=$(date +%s)
    time python "$@"
    end_time=$(date +%s)
    echo "Execution time: $((end_time - start_time)) seconds"
    echo -e "\n----------------------------------------\n"
}

# Run each script and print the output

run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Ackley --dim 4 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Beale --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Branin --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Bukin --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Cosine8 --dim 8 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function DixonPrice --dim 3 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function DropWave --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Griewank --dim 5 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Hartmann --dim 6 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Levy --dim 3 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Michalewicz --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Rastrigin --dim 3 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Rosenbrock --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function SixHumpCamel --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function ThreeHumpCamel --dim 2 --kernel_weight_type uniform --acq_weight random
run_and_print MMMA.py --experiments 1 --seed 101 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Shekel --dim 4 --kernel_weight_type uniform --acq_weight random

echo "Execution complete."
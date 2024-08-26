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
run_and_print MMMA.py --experiments 25 --seed 0 --function SixHumpCamel --dim 2 --kernels Matern52 Matern32 RBF --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function ThreeHumpCamel --dim 2 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 100 --function Hartmann --dim 6 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Shekel --dim 4 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Rosenbrock --dim 2 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Ackley --dim 4 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Rastrigin --dim 3 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Michalewicz --dim 2 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Griewank --dim 5 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Cosine8 --dim 8 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function HolderTable --dim 2 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform
run_and_print MMMA.py --experiments 25 --seed 0 --function Levy --dim 3 --kernels Matern52 Matern32 RBF  --acquisition LogEI LogPI UCB --acq_weight random --kernel_weight_type uniform

echo "Execution complete."
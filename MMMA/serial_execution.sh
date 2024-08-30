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


# run_and_print MMMA.py --experiments 5 --seed 100 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function DropWave --dim 2 --kernel_weight_type likelihood --acq_weight random
run_and_print MMMA.py --experiments 5 --seed 300 --acquisition LogEI LogPI UCB --kernels Matern52 Matern32 RBF --function Michalewicz --dim 2 --kernel_weight_type likelihood --acq_weight random


echo "Execution complete."
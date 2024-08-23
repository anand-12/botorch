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
run_and_print GP_Hedge.py --experiments 25 --seed 0 --function Ackley --dim 4 --kernel Matern52 --acquisition LogEI LogPI UCB

run_and_print multi_model_single_acqu.py --experiments 25 --seed 0 --function Ackley --dim 4 --kernel RBF Matern32 Matern52 --acquisition LogPI

run_and_print MMMA.py --experiments 25 --seed 0 --function Ackley --dim 4 --kernel Matern52 RBF Matern32 --acquisition LogEI LogPI UCB

echo "Execution complete."
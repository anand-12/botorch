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

run_and_print PI_is_back.py --seed 0 --kernel Matern52 --experiments 10 --switch_percentage 25 --function ThreeHumpCamel --dim 4 


echo "Execution complete."
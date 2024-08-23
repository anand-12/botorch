#!/bin/bash

run_python_script() {
    python $1 "${@:2}"
}

# "script_name.py arguments"
args_list=(
    "baseline.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel Matern52 --acquisition LogEI"
    "GP_Hedge.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel Matern52 --acquisition LogEI LogPI UCB"
    "multi_model_single_acqu.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel RBF Matern32 Matern52 --acquisition LogPI"
    "multi_model_single_acqu.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel RBF Matern32 Matern52 --acquisition LogPI --weight_type likelihood"
    "multi_model_single_acqu.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel RBF Matern32 Matern52 --acquisition LogPI --true_ensemble"
    "multi_model_single_acqu.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel RBF Matern32 Matern52 --acquisition LogPI --true_ensemble --weight_type likelihood"
    "MMMA.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel Matern52 RBF Matern32 --acquisition LogEI LogPI UCB"
    "MMMA.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel Matern52 RBF Matern32 --acquisition LogEI LogPI UCB --weight_type likelihood"
    "MMMA.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel Matern52 RBF Matern32 --acquisition LogEI LogPI UCB --true_ensemble"
    "MMMA.py --experiments 10 --seed 0 --function Hartmann --dim 6 --kernel Matern52 RBF Matern32 --acquisition LogEI LogPI UCB --true_ensemble --weight_type likelihood"
)

max_jobs=5

for args in "${args_list[@]}"; do
    while (( $(jobs -p | wc -l) >= max_jobs )); do
        sleep 1
    done
    run_python_script $args &
done

wait
echo "All jobs completed."
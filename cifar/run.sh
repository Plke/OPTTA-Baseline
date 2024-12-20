#!/bin/bash

set -Eeuxo pipefail

while getopts ":g:d:" opt; do
    # shellcheck disable=SC2220
    case "$opt" in
        g)
            gpu="$OPTARG"
            ;;
        d)
            dataset="$OPTARG"
            ;;
    esac
done

for adaptation in source norm cotta; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --dataset $dataset --save_dir "./output/${dataset}/${adaptation}"
done

# Tent, EATA, OSTTA
for adaptation in tent eata ostta; do
    for alpha in 1.0 0.5 0.2 0.1; do
        CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --dataset $dataset --save_dir "./output/${dataset}/${adaptation}" --alpha $alpha
    done
done

# Tent, EATA, OSTTA + UniEnt, UniEnt+
for adaptation in tent eata ostta; do
    for alpha1 in 1.0 0.5 0.2 0.1; do
        for alpha2 in 1.0 0.5 0.2 0.1; do 
            for criterion in ent_ind_ood ent_unf; do # uni uni+
                CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --dataset $dataset --save_dir "./output/${dataset}/${adaptation}" --alpha $alpha1 $alpha2 --criterion $criterion
            done
        done
    done
done

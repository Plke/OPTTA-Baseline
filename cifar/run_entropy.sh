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


# adaptation=ostta_neigh
# alpha1=0.5
# alpha2=0
# batch_size=100

# CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation  --dataset $dataset --save_dir "./output/${dataset}/${adaptation}_${batch_size}" --alpha $alpha1 $alpha2 --batch_size $batch_size 

for adaptation in entropy ; do
    for thresh_hold in 0.85 0.88 0.92 0.94 0.96 0.98;do 
        for alpha1 in 1.0 0.5 0.2 0.1 0; do
            for alpha2 in 1.0 0.5 0.2 0.1 0; do 
                CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation  --dataset $dataset --save_dir "./output/${dataset}/${adaptation}" --alpha $alpha1 $alpha2  --thresh_hold $thresh_hold
            done
        done
    done
done




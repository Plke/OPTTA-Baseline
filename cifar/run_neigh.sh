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

for adaptation in ostta_neigh ; do
    # for nr in 1 2 3 4 5;do 
        for batch_size in 100  ; do
            for alpha1 in 1.0 0.5 0.2 0.1 0; do
                for alpha2 in 1.0 0.5 0.2 0.1 0; do 
                    CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation  --dataset $dataset --save_dir "./output/${dataset}/${adaptation}_only_${batch_size}" --alpha $alpha1 $alpha2 --batch_size $batch_size 
                done
            done
        done
    # done
done




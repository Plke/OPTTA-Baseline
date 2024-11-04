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



for adaptation in ostta_neigh ; do
for batch_size in 100 200 400; do
    for alpha1 in 1.0 0.5 0.2 0.1 0; do
        for alpha2 in 1.0 0.5 0.2 0.1 0; do 
            CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation  --dataset $dataset --save_dir "./output/${dataset}/${adaptation}" --alpha $alpha1 $alpha2 --batch_size $batch_size

        done
    done
    done
done




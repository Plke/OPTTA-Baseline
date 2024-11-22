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

for adaptation in oslpp; do
    for alpha1 in 1.0 0.5 0.1 0; do
        for alpha2 in 1.0 0.5 0.1 0; do
            for nr in 10 20 50; do
                    CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --dataset $dataset --save_dir "./output/oslpp" --alpha $alpha1 $alpha2  --nr $nr
            done
        done
    done
done


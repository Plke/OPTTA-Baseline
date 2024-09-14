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



# Tent, EATA, OSTTA
for gamma in 1 0.9 0.95 0.99 0.995 0.999 ; do
  CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation ostta_ema --gamma $gamma --dataset $dataset --save_dir "./output/ema" 
done

CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation ostta --dataset $dataset --save_dir "./output/ema" 


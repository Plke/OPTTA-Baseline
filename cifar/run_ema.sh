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
for adaptation in ostta ostta_ema; do
  CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --dataset $dataset --save_dir "./output/ema" 
done



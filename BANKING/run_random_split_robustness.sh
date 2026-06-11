#!/bin/bash

set -e

CLASS_SPLIT_SEEDS=(202 303)
DATA_SEEDS=(43 44)

for class_seed in "${CLASS_SPLIT_SEEDS[@]}"
do
  for data_seed in "${DATA_SEEDS[@]}"
  do
    echo "================================================="
    echo "Random split robustness run"
    echo "Class split seed: ${class_seed}, Data seed: ${data_seed}"
    echo "================================================="

    python3 -u head_tail_split_robuse.py \
      --seed ${data_seed} \
      --split-mode random \
      --class-split-seed ${class_seed} \
      --batch-size 40 \
      --n-iterations 40 \
      --plateau-delta 0.005 \
      --plateau-patience 3
  done
done
#!/usr/bin/env bash
#
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run.sh 1 2 > /dev/null 2>&1 &   # toy + mnist
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run.sh 3 4 5 > /dev/null 2>&1 & # pathmnist_c1/c2 + sicap + T-ablation
# nohup bash scripts/run.sh 0 > /dev/null 2>&1 &                             # baselines knn/mah (CPU)
# SUMMARIZE=1 bash scripts/run.sh                                             # resumen final

set -euo pipefail
DIR="$(dirname "$0")"
source "$DIR/core.sh"
source "$DIR/experiments.sh"
gpu_select && mkdir -p results/summary
for id in "$@"; do "exp_$id"; done
[[ "${SUMMARIZE:-0}" == "1" ]] && summary

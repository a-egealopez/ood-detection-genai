#!/usr/bin/env bash
#
# 2-GPU setup — mnist y pathmnist son los más lentos, se reparten entre GPUs:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run.sh 1 2 > /dev/null 2>&1 &      # toy (rápido) + mnist (lento)
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run.sh 3 4 5 > /dev/null 2>&1 &    # pathmnist (lento) + sicap + T-ablation
#   nohup bash scripts/run.sh 0 > /dev/null 2>&1 &                               # baseline en CPU (ligero)
#   al terminar: SUMMARIZE=1 bash scripts/run.sh                                 # resumen final

set -euo pipefail
DIR="$(dirname "$0")"
source "$DIR/core.sh"
source "$DIR/experiments.sh"
gpu_select && mkdir -p results/summary
for id in "$@"; do "exp_$id"; done
[[ "${SUMMARIZE:-0}" == "1" ]] && summary

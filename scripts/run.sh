#!/usr/bin/env bash
#
# 2-GPU setup — mnist y pathmnist son los más lentos, se reparten entre GPUs:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh 1 2               # toy (rápido) + mnist (lento)
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh 3 4 5             # pathmnist (lento) + sicap + T-ablation
#   bash scripts/run.sh 0                                        # baseline en CPU (ligero)
#   al terminar: SUMMARIZE=1 bash scripts/run.sh                 # resumen final

set -euo pipefail
DIR="$(dirname "$0")"
source "$DIR/core.sh"
source "$DIR/experiments.sh"
gpu_select && mkdir -p results/logs results/summary
for id in "$@"; do "exp_$id"; done
[[ "${SUMMARIZE:-0}" == "1" ]] && summary

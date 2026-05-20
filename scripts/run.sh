#!/usr/bin/env bash
#
# 2-GPU setup (terminal por línea, lanzar a la vez):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh 1 2               # toy + mnist en GPU 0
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh 3 4                # sicap en GPU 1
#   bash scripts/run.sh 0                                        # baseline en CPU
#   al terminar las tres: SUMMARIZE=1 bash scripts/run.sh        # resumen final

set -euo pipefail
DIR="$(dirname "$0")"
source "$DIR/core.sh"
source "$DIR/experiments.sh"
gpu_select && mkdir -p results/logs results/summary
for id in "$@"; do "exp_$id"; done
[[ "${SUMMARIZE:-0}" == "1" ]] && summary

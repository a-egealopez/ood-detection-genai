#!/usr/bin/env bash
#
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run.sh 1 2 > /dev/null 2>&1 &   # toy + mnist
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run.sh 3 4 5 > /dev/null 2>&1 & # pathmnist_c1/c2 + sicap + T-ablation
# nohup bash scripts/run.sh 0 > /dev/null 2>&1 &                             # baselines knn/mah (CPU)
# SUMMARIZE=1 bash scripts/run.sh                                          # resumen final

## for i in {0..4}; do nohup bash scripts/run.sh "$i" > "logs/log_$i.txt" 2>&1 & sleep 1; done; wait; nohup bash scripts/run.sh 5 > "logs/log_5.txt" 2>&1; SUMMARIZE=1 nohup bash scripts/run.sh > "logs/log_summary.txt" 2>&1 &
## pkill -u alejegealopez python3

set -euo pipefail
DIR="$(dirname "$0")"

# En contextos no-interactivos (nohup) ~/.bashrc no se ejecuta.
# Leemos WANDB_API_KEY desde ~/.netrc para que wandb funcione en background.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    _wandb_key=$(python3 -c \
        "import netrc, sys; a=netrc.netrc().authenticators('api.wandb.ai'); print(a[2] if a else '')" \
        2>/dev/null || true)
    [[ -n "${_wandb_key:-}" ]] && export WANDB_API_KEY="$_wandb_key"
    unset _wandb_key
fi

source "$DIR/core.sh"
source "$DIR/experiments.sh"
gpu_select && mkdir -p results/summary
for id in "$@"; do "exp_$id"; done
[[ "${SUMMARIZE:-0}" == "1" ]] && summary

#!/usr/bin/env bash

SEED="${SEED:-42}"
TOY_DATASETS="${TOY_DATASETS:-moons blobs}"
SICAP_DATASETS="${SICAP_DATASETS:-sicap_c1 sicap_c12}"
SEED_TOY="${SEED_TOY:-42}"
LR_VAE_TOY="${LR_VAE_TOY:-1e-3}"
LR_DDPM_TOY="${LR_DDPM_TOY:-1e-3}"

SEED_MNIST="${SEED_MNIST:-42}"
LR_VAE_MNIST="${LR_VAE_MNIST:-1e-4}"
LR_DDPM_MNIST="${LR_DDPM_MNIST:-1e-4}"

SEED_PATH="${SEED_PATH:-42}"
LR_VAE_PATH="${LR_VAE_PATH:-1e-4}"
LR_DDPM_PATH="${LR_DDPM_PATH:-1e-4}"

SEEDS="${SEEDS:-42 107 2024}"
LRS_VAE="${LRS_VAE:-1e-4 5e-4 1e-3}"
LRS_DDPM="${LRS_DDPM:-1e-4 5e-4 1e-3}"


T_VALUES="${T_VALUES:-5 10 25 50}"
BEST_SEED="${BEST_SEED:-42}"
BEST_LR_DDPM="${BEST_LR_DDPM:-1e-4}"

LOGS_DIR="${LOGS_DIR:-results}"
OUT_CSV="${OUT_CSV:-results/summary/comparison.csv}"

# --- utilities ---
gpu_select() {
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
            --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 \
            | cut -d',' -f1 | tr -d ' ')
        export CUDA_VISIBLE_DEVICES
        echo "[gpu] selected device $CUDA_VISIBLE_DEVICES"
    fi
}

log_info() { echo "[info] $*"; }
log_ok()   { echo "[done] $*"; }

log_time() {
    local label=$1 start=$2 end=$3
    local elapsed=$((end - start))
    printf "  [time] %s -> %02d:%02d:%02d\n" "$label" \
        $((elapsed/3600)) $(( (elapsed/60)%60 )) $((elapsed%60))
}

run_python() { python3 main.py "$@" 2>&1 | sed 's/^/  /'; }

run_timed() {
    local label=$1; shift
    local t0; t0=$(date +%s)
    run_python "$@"
    log_time "$label" "$t0" "$(date +%s)"
}

# run_loop SKIP_FLAG EXPERIMENT DATASET SEEDS_STR LRS_STR
run_loop() {
    local skip_flag=$1 exp=$2 dataset=$3 seeds=$4 lrs=$5
    for seed in $seeds; do
        for lr in $lrs; do
            log_info "[$exp] $dataset seed=$seed lr=$lr"
            run_timed "$exp/$dataset/s$seed/lr$lr" \
                --mode reconstruction-method \
                --experiment "$exp" --dataset "$dataset" \
                --lr "$lr" --seed "$seed" "$skip_flag"
        done
    done
}

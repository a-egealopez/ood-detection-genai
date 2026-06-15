#!/usr/bin/env bash

TOY_DATASETS="${TOY_DATASETS:-moons blobs}"
SICAP_DATASETS="${SICAP_DATASETS:-sicap_c1 sicap_c12}"
PATH_DATASETS="${PATH_DATASETS:-pathmnist_c1 pathmnist_c2}"

SEED="${SEED:-42}"

SEED_TOY="${SEED_TOY:-42}"
LR_VAE_TOY="${LR_VAE_TOY:-1e-3}"
LR_DDPM_TOY="${LR_DDPM_TOY:-1e-3}"

SEED_MNIST="${SEED_MNIST:-42}"
LR_MLP_MNIST="${LR_MLP_MNIST:-1e-4}"
LR_MLP_PATH="${LR_MLP_PATH:-1e-4}"

SEEDS="${SEEDS:-42 107 2024}"
LRS_VAE="${LRS_VAE:-1e-4 5e-4 1e-3}"
LRS_DDPM="${LRS_DDPM:-1e-4 5e-4 1e-3}"

SEEDS_PATH="${SEEDS_PATH:-42 107 2024}"
LRS_VAE_PATH="${LRS_VAE_PATH:-1e-4 5e-4 1e-3}"
LRS_DDPM_PATH="${LRS_DDPM_PATH:-1e-4 5e-4 1e-3}"

SEEDS_UNET="${SEEDS_UNET:-42}"
LRS_UNET_VAE="${LRS_UNET_VAE:-1e-4 5e-4 1e-3}"
LRS_UNET_DDPM="${LRS_UNET_DDPM:-1e-4}"

T_VALUES="${T_VALUES:-5 10 25 50}"

BEST_SEED="${BEST_SEED:-42}"
BEST_LR_DDPM="${BEST_LR_DDPM:-1e-3}"

BEST_SEED_PATH="${BEST_SEED_PATH:-42}"
BEST_LR_DDPM_PATH="${BEST_LR_DDPM_PATH:-1e-4}"

BEST_SEED_MNIST="${BEST_SEED_MNIST:-42}"
BEST_LR_DDPM_MNIST="${BEST_LR_DDPM_MNIST:-1e-4}"

LOGS_DIR="${LOGS_DIR:-results/evaluation}"
OUT_CSV="${OUT_CSV:-results/summary/comparison.csv}"

MODEL=${MODEL:-both}
SKIP_SCORES="${SKIP_SCORES:-}"

should_run() {
    local target_model=$1
    [[ "$MODEL" == "both" || "$MODEL" == "$target_model" ]]
}

gpu_select() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[gpu] using preset device $CUDA_VISIBLE_DEVICES"
        return
    fi
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

run_python() { python3 main.py "$@" 2>&1 | stdbuf -oL sed 's/^/  /'; }

run_timed() {
    local label=$1; shift
    local t0; t0=$(date +%s)
    run_python "$@"
    log_time "$label" "$t0" "$(date +%s)"
}

run_loop() {
    local skip_flag=$1 exp=$2 dataset=$3 seeds=$4 lrs=$5
    local extra_args=()
    [[ -n "$SKIP_SCORES" ]] && extra_args=(--skip-scores $SKIP_SCORES)
    for seed in $seeds; do
        for lr in $lrs; do
            log_info "[$exp] $dataset seed=$seed lr=$lr"
            run_timed "$exp/$dataset/s$seed/lr$lr" \
                --mode reconstruction-method \
                --experiment "$exp" --dataset "$dataset" \
                --lr "$lr" --seed "$seed" "$skip_flag" \
                ${extra_args[@]+"${extra_args[@]}"}
        done
    done
}

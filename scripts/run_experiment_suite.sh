#!/usr/bin/env bash

# Terminal 1+2
#CUDA_VISIBLE_DEVICES=1 MODELS="vae" ./scripts/run_experiment_suite.sh
#CUDA_VISIBLE_DEVICES=1 RUN_DISTANCE=0 MODELS="ddpm" ./scripts/run_experiment_suite.sh
#watch -n 2 nvidia-smi

# MODELS="ddpm" SKIP_TRAIN=1 RUN_DISTANCE=0 ./scripts/run_experiment_suite.sh

set -euo pipefail

DATASETS="${DATASETS:-mnist sicap}"
SEEDS="${SEEDS:-42 107}"
LRS="${LRS:-1e-3 1e-4}"
MODELS="${MODELS:-vae ddpm}"
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-1}"
RUN_DISTANCE="${RUN_DISTANCE:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Usando GPU $CUDA_VISIBLE_DEVICES"

mkdir -p results/logs results/summary

log_info() { echo " $*"; }
log_success() { echo " $*"; }

log_time() {
    local label=$1
    local start=$2
    local end=$3
    local elapsed=$((end - start))
    printf "%s -> %02d:%02d:%02d\n" "$label" \
        $((elapsed/3600)) $(((elapsed/60)%60)) $((elapsed%60))
}

#########################################################
# Reconstruction-based Methods
#########################################################
run_reconstruction_methods() {
    local timing_file="results/summary/timing_reconstruction.txt"
    {
        echo "Reconstruction-based OOD Detection"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
    } > "$timing_file"

    log_info "Starting reconstruction-based methods..."

    local total_runs=0
    local current_run=0

    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            for lr in $LRS; do
                for model in $MODELS; do
                    ((total_runs++))
                done
            done
        done
    done

    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            for lr in $LRS; do
                for model in $MODELS; do
                    ((current_run++))

                    log_info "[$model] Dataset: $dataset, LR: $lr, Seed: $seed (${current_run}/${total_runs})"

                    local start_time=$(date +%s)

                    local extra_args=()
                    if [[ "$SKIP_TRAIN" == "1" ]]; then
                        extra_args+=(--skip-train)
                    fi

                    python3 main.py \
                        --mode reconstruction-method \
                        --experiment "$model" \
                        --dataset "$dataset" \
                        --lr "$lr" \
                        --seed "$seed" \
                        "${extra_args[@]}" \
                        2>&1 | sed 's/^/    /'

                    local end_time=$(date +%s)
                    log_time "  $model/$dataset/lr${lr}/seed$seed" "$start_time" "$end_time" \
                        | tee -a "$timing_file"
                done
            done
        done
    done

    log_success "Reconstruction methods completed"
}

#########################################################
# Distance-based Methods
#########################################################
run_distance_methods() {
    local timing_file="results/summary/timing_distance.txt"
    {
        echo "Distance-based OOD Detection (KNN, Residual)"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"
    } > "$timing_file"

    log_info "Starting distance-based methods..."

    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            for distance_type in knn residual; do

                log_info "[${distance_type^^}] Dataset: $dataset, Seed: $seed"

                local start_time=$(date +%s)

                python3 main.py \
                    --mode distance-method \
                    --dataset "$dataset" \
                    --distance-type "$distance_type" \
                    --seed "$seed" \
                    --config configs/base \
                    2>&1 | sed 's/^/    /'

                local end_time=$(date +%s)
                log_time "  $distance_type/$dataset/seed$seed" "$start_time" "$end_time" \
                    | tee -a "$timing_file"
            done
        done
    done

    log_success "Distance methods completed"
}

#########################################################
# Summary
#########################################################
generate_summary() {
    log_info "Generating summary statistics..."

    python3 main.py \
        --mode stats-summary \
        --logs-dir "results/logs" \
        --out-csv "results/summary/comparison.csv" \
        2>&1 | sed 's/^/  /'

    log_success "Summary generated"
}

#########################################################
# Main
#########################################################
main() {
    echo ""
    echo "╔═════════════════════════════════════════════════════╗"
    echo "║  OOD Detection GenAI - Comparison Suite             ║"
    echo "╚═════════════════════════════════════════════════════╝"
    echo ""

    [[ "$RUN_RECONSTRUCTION" == "1" ]] && run_reconstruction_methods && echo ""
    [[ "$RUN_DISTANCE" == "1" ]] && run_distance_methods && echo ""

    generate_summary
}

main
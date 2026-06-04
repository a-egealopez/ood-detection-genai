#!/usr/bin/env bash
# Numbered experiment functions — source this file, do not execute directly.
#
# Convention:
#   exp_N_train / exp_N_eval  — granular steps
#   exp_N                     — calls train then eval (or standalone for 0/4/5)
#
# Experiments:
#   0  — distance baselines (knn + mahalanobis) for sicap + pathmnist
#   1  — toy (moons + blobs)
#   2  — MNIST
#   3  — PathMNIST key (pathmnist_c1 + pathmnist_c2, multi-seed + multi-lr)
#   4  — SICAP key     (sicap_c1    + sicap_c12,    multi-seed + multi-lr)
#   5  — T ablation    (ddpm, sicap + pathmnist, best seed + lr)


# ── 0. Distance baselines (knn + mahalanobis) ─────────────────────────────────

exp_0() {
    log_info "[0] distance baseline (knn + mahalanobis)"

    log_info "[0] sicap (seed=$SEED)"
    for dataset in $SICAP_DATASETS; do
        for dtype in knn mahalanobis; do
            log_info "[dist-$dtype] $dataset seed=$SEED"
            run_timed "dist-$dtype/$dataset" \
                --mode distance-method \
                --experiment vae --dataset "$dataset" \
                --distance-type "$dtype" --seed "$SEED"
        done
    done

    log_info "[0] pathmnist (seed=$SEED)"
    for dataset in $PATH_DATASETS; do
        for dtype in knn mahalanobis; do
            log_info "[dist-$dtype] $dataset seed=$SEED"
            run_timed "dist-$dtype/$dataset" \
                --mode distance-method \
                --experiment vae_path --dataset "$dataset" \
                --distance-type "$dtype" --seed "$SEED"
        done
    done
}

# ── 1. Toy (moons + blobs) ────────────────────────────────────────────────────

exp_1_train() {
    log_info "[1_train] toy train (datasets=(moons + blobs))"
    for dataset in $TOY_DATASETS; do
        should_run vae && run_loop --skip-eval vae_toy  "$dataset" "$SEED_TOY" "$LR_VAE_TOY"
        should_run ddpm && run_loop --skip-eval ddpm_toy "$dataset" "$SEED_TOY" "$LR_DDPM_TOY"
    done
}

exp_1_eval() {
    log_info "[1_eval] toy eval (datasets=(moons + blobs))"
    for dataset in $TOY_DATASETS; do
        should_run vae && run_loop --skip-train vae_toy  "$dataset" "$SEED_TOY" "$LR_VAE_TOY"
        should_run ddpm && run_loop --skip-train ddpm_toy "$dataset" "$SEED_TOY" "$LR_DDPM_TOY"
    done
}

exp_1() { exp_1_train; exp_1_eval; }

# ── 2. MNIST ──────────────────────────────────────────────────────────────────

exp_2_train() {
    log_info "[2_train] mnist train"
    should_run vae && run_loop --skip-eval vae  mnist "$SEED_MNIST" "$LR_VAE_MNIST"
    should_run ddpm && run_loop --skip-eval ddpm mnist "$SEED_MNIST" "$LR_DDPM_MNIST"
}

exp_2_eval() {
    log_info "[2_eval] mnist eval"
    should_run vae && run_loop --skip-train vae  mnist "$SEED_MNIST" "$LR_VAE_MNIST"
    should_run ddpm && run_loop --skip-train ddpm mnist "$SEED_MNIST" "$LR_DDPM_MNIST"
}

exp_2() { exp_2_train; exp_2_eval; }

# ── 3. PathMNIST key (pathmnist_c1 + pathmnist_c2, multi-seed + multi-lr) ─────

exp_3_train() {
    log_info "[3_train] pathmnist_c1/c2 train (seeds=$SEEDS_PATH lrs_vae=$LRS_VAE_PATH lrs_ddpm=$LRS_DDPM_PATH)"
    for dataset in $PATH_DATASETS; do
        should_run vae && run_loop --skip-eval vae_path  "$dataset" "$SEEDS_PATH" "$LRS_VAE_PATH"
        should_run ddpm && run_loop --skip-eval ddpm_path "$dataset" "$SEEDS_PATH" "$LRS_DDPM_PATH"
    done
}

exp_3_eval() {
    log_info "[3_eval] pathmnist_c1/c2 eval (seeds=$SEEDS_PATH)"
    for dataset in $PATH_DATASETS; do
        should_run vae && run_loop --skip-train vae_path  "$dataset" "$SEEDS_PATH" "$LRS_VAE_PATH"
        should_run ddpm && run_loop --skip-train ddpm_path "$dataset" "$SEEDS_PATH" "$LRS_DDPM_PATH"
    done
}

exp_3() { exp_3_train; exp_3_eval; }

# ── 4. SICAP key (sicap_c1 + sicap_c12, multi-seed + multi-lr) ───────────────

exp_4_train() {
    log_info "[4_train] sicap train (datasets=sicap_c1 sicap_c12 seeds=$SEEDS)"
    for dataset in $SICAP_DATASETS; do
        should_run vae && run_loop --skip-eval vae  "$dataset" "$SEEDS" "$LRS_VAE"
        should_run ddpm && run_loop --skip-eval ddpm "$dataset" "$SEEDS" "$LRS_DDPM"
    done
}

exp_4_eval() {
    log_info "[4_eval] sicap eval (datasets=sicap_c1 sicap_c12 seeds=$SEEDS)"
    for dataset in $SICAP_DATASETS; do
        should_run vae && run_loop --skip-train vae  "$dataset" "$SEEDS" "$LRS_VAE"
        should_run ddpm && run_loop --skip-train ddpm "$dataset" "$SEEDS" "$LRS_DDPM"
    done
}

exp_4() { exp_4_train; exp_4_eval; }

# ── 5. T ablation (ddpm, sicap + pathmnist, best seed + lr) ──────────────────

exp_5() {
    log_info "[5] ddpm T ablation — sicap (seed=$BEST_SEED lr=$BEST_LR_DDPM T=$T_VALUES)"
    for dataset in $SICAP_DATASETS; do
        for t in $T_VALUES; do
            log_info "[ddpm] $dataset T=$t"
            run_timed "ddpm/$dataset/T$t" \
                --mode reconstruction-method \
                --experiment ddpm --dataset "$dataset" \
                --lr "$BEST_LR_DDPM" --seed "$BEST_SEED" \
                --n-score-steps "$t" --skip-train
        done
    done

    log_info "[5] ddpm T ablation — pathmnist (seed=$BEST_SEED_PATH lr=$BEST_LR_DDPM_PATH T=$T_VALUES)"
    for dataset in $PATH_DATASETS; do
        for t in $T_VALUES; do
            log_info "[ddpm_path] $dataset T=$t"
            run_timed "ddpm_path/$dataset/T$t" \
                --mode reconstruction-method \
                --experiment ddpm_path --dataset "$dataset" \
                --lr "$BEST_LR_DDPM_PATH" --seed "$BEST_SEED_PATH" \
                --n-score-steps "$t" --skip-train
        done
    done
}

# ── summary (not an experiment — call explicitly when needed) ─────────────────

summary() {
    log_info "generating summary"
    run_python --mode stats-summary --logs-dir "$LOGS_DIR" --out-csv "$OUT_CSV"
    log_ok "summary saved to $OUT_CSV"
}

# ── all ───────────────────────────────────────────────────────────────────────

exp_all() {
    exp_0
    exp_1
    exp_2
    exp_3
    exp_4
    exp_5
    summary
}

exp_all_eval() {
    exp_0
    exp_1_eval
    exp_2_eval
    exp_3_eval
    exp_4_eval
    exp_5
    summary
}

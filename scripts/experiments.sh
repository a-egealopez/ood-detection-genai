#!/usr/bin/env bash
# Numbered experiment functions — source this file, do not execute directly.
#
# Convention:
#   exp_N_train / exp_N_eval  — granular steps
#   exp_N                     — calls train then eval (or standalone for 0/4)

# ── 0. Distance baseline (knn + mahalanobis, sicap) ──────────────────────────

exp_0() {
    log_info "[0] distance baseline (knn + mahalanobis, seed=$SEED)"
    for dataset in $SICAP_DATASETS; do
        for dtype in knn mahalanobis; do
            log_info "[dist-$dtype] $dataset seed=$SEED"
            run_timed "dist-$dtype/$dataset" \
                --mode distance-method \
                --experiment vae --dataset "$dataset" \
                --distance-type "$dtype" --seed "$SEED"
        done
    done
}

# ── 1. Toy (moons + blobs) ─────────────────────────────────────────────────

exp_1_train() {
    log_info "[1_train] toy train (datasets=(moons + blobs))"
    for dataset in $TOY_DATASETS; do
        run_loop --skip-eval vae_toy  "$dataset" "$SEED_TOY" "$LR_VAE_TOY"
        run_loop --skip-eval ddpm_toy "$dataset" "$SEED_TOY" "$LR_DDPM_TOY"
    done
}

exp_1_eval() {
    log_info "[1_eval] toy eval (datasets=(moons + blobs))"
    for dataset in $TOY_DATASETS; do
        run_loop --skip-train vae_toy  "$dataset" "$SEED_TOY" "$LR_VAE_TOY"
        run_loop --skip-train ddpm_toy "$dataset" "$SEED_TOY" "$LR_DDPM_TOY"
    done
}

exp_1() { exp_1_train; exp_1_eval; }

# ── 2. MNIST ─────────────────────────────────────────────────────────────────

exp_2_train() {
    log_info "[2_train] mnist train"
    run_loop --skip-eval vae  mnist "$SEED_MNIST" "$LR_VAE_MNIST"
    run_loop --skip-eval ddpm mnist "$SEED_MNIST" "$LR_DDPM_MNIST"
}

exp_2_eval() {
    log_info "[2_eval] mnist eval"
    run_loop --skip-train vae  mnist "$SEED_MNIST" "$LR_VAE_MNIST"
    run_loop --skip-train ddpm mnist "$SEED_MNIST" "$LR_DDPM_MNIST"
}

exp_2() { exp_2_train; exp_2_eval; }

# ── 3. PathMNIST (NORM vs TUM) ───────────────────────────────────────────────

exp_3_train() {
    log_info "[3_train] pathmnist train (seed=$SEED_PATH)"
    run_loop --skip-eval vae_path  pathmnist "$SEED_PATH" "$LR_VAE_PATH"
    run_loop --skip-eval ddpm_path pathmnist "$SEED_PATH" "$LR_DDPM_PATH"
}

exp_3_eval() {
    log_info "[3_eval] pathmnist eval (seed=$SEED_PATH)"
    run_loop --skip-train vae_path  pathmnist "$SEED_PATH" "$LR_VAE_PATH"
    run_loop --skip-train ddpm_path pathmnist "$SEED_PATH" "$LR_DDPM_PATH"
}

exp_3() { exp_3_train; exp_3_eval; }

# ── 4. SICAP (sicap_c1 + sicap_c12) ─────────────────────────────────────────

exp_4_train() {
    log_info "[4_train] sicap train (datasets=sicap_c1 sicap_c12 seeds=$SEEDS)"
    for dataset in $SICAP_DATASETS; do
        run_loop --skip-eval vae  "$dataset" "$SEEDS" "$LRS_VAE"
        run_loop --skip-eval ddpm "$dataset" "$SEEDS" "$LRS_DDPM"
    done
}

exp_4_eval() {
    log_info "[4_eval] sicap eval (datasets=sicap_c1 sicap_c12 seeds=$SEEDS)"
    for dataset in $SICAP_DATASETS; do
        run_loop --skip-train vae  "$dataset" "$SEEDS" "$LRS_VAE"
        run_loop --skip-train ddpm "$dataset" "$SEEDS" "$LRS_DDPM"
    done
}

exp_4() { exp_4_train; exp_4_eval; }

# ── 5. T ablation (ddpm, best seed + lr) ─────────────────────────────────────

exp_5() {
    log_info "[5] ddpm T ablation (seed=$BEST_SEED lr=$BEST_LR_DDPM T=$T_VALUES)"
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

#!/usr/bin/env bash

exp_0() {
    for dataset in $SICAP_DATASETS; do
        for dtype in knn mahalanobis; do
            run_timed "dist-$dtype/$dataset" \
                --mode distance-method \
                --experiment "distance/sicap" --dataset "$dataset" \
                --distance-type "$dtype" --seed "$SEED"
        done
    done
    for dataset in $PATH_DATASETS; do
        for dtype in knn mahalanobis; do
            run_timed "dist-$dtype/$dataset" \
                --mode distance-method \
                --experiment "distance/path" --dataset "$dataset" \
                --distance-type "$dtype" --seed "$SEED"
        done
    done
}

exp_1_train() {
    for dataset in $TOY_DATASETS; do
        should_run vae  && run_loop --skip-eval "mlp/vae/toy"  "$dataset" "$SEED_TOY" "$LR_VAE_TOY"
        should_run ddpm && run_loop --skip-eval "mlp/ddpm/toy" "$dataset" "$SEED_TOY" "$LR_DDPM_TOY"
    done
}

exp_1_eval() {
    for dataset in $TOY_DATASETS; do
        should_run vae  && run_loop --skip-train "mlp/vae/toy"  "$dataset" "$SEED_TOY" "$LR_VAE_TOY"
        should_run ddpm && run_loop --skip-train "mlp/ddpm/toy" "$dataset" "$SEED_TOY" "$LR_DDPM_TOY"
    done
}

exp_1() { exp_1_train; exp_1_eval; }

exp_2_train() {
    should_run vae  && run_loop --skip-eval "mlp/vae/sicap"  mnist        "$SEED_MNIST" "$LR_MLP_MNIST"
    should_run ddpm && run_loop --skip-eval "mlp/ddpm/mnist" mnist        "$SEED_MNIST" "$LR_MLP_MNIST"
    should_run vae  && run_loop --skip-eval "mlp/vae/path"   pathmnist_c1 "$SEED_MNIST" "$LR_MLP_PATH"
    should_run ddpm && run_loop --skip-eval "mlp/ddpm/path"  pathmnist_c1 "$SEED_MNIST" "$LR_MLP_PATH"
}

exp_2_eval() {
    should_run vae  && run_loop --skip-train "mlp/vae/sicap"  mnist        "$SEED_MNIST" "$LR_MLP_MNIST"
    should_run ddpm && run_loop --skip-train "mlp/ddpm/mnist" mnist        "$SEED_MNIST" "$LR_MLP_MNIST"
    should_run vae  && run_loop --skip-train "mlp/vae/path"   pathmnist_c1 "$SEED_MNIST" "$LR_MLP_PATH"
    should_run ddpm && run_loop --skip-train "mlp/ddpm/path"  pathmnist_c1 "$SEED_MNIST" "$LR_MLP_PATH"
}

exp_2() { exp_2_train; exp_2_eval; }

exp_3_train() {
    should_run vae  && run_loop --skip-eval "unet/vae/base"  mnist "$SEEDS_UNET" "$LRS_UNET"
    should_run ddpm && run_loop --skip-eval "unet/ddpm/base" mnist "$SEEDS_UNET" "$LRS_UNET"
}

exp_3_eval() {
    should_run vae  && run_loop --skip-train "unet/vae/base"  mnist "$SEEDS_UNET" "$LRS_UNET"
    should_run ddpm && run_loop --skip-train "unet/ddpm/base" mnist "$SEEDS_UNET" "$LRS_UNET"
}

exp_3() { exp_3_train; exp_3_eval; }

exp_4_train() {
    for dataset in $PATH_DATASETS; do
        should_run vae  && run_loop --skip-eval "unet/vae/base"  "$dataset" "$SEEDS_UNET" "$LRS_UNET_VAE"
        should_run ddpm && run_loop --skip-eval "unet/ddpm/base" "$dataset" "$SEEDS_UNET" "$LRS_UNET_DDPM"
    done
}

exp_4_eval() {
    for dataset in $PATH_DATASETS; do
        should_run vae  && run_loop --skip-train "unet/vae/base"  "$dataset" "$SEEDS_UNET" "$LRS_UNET_VAE"
        should_run ddpm && run_loop --skip-train "unet/ddpm/base" "$dataset" "$SEEDS_UNET" "$LRS_UNET_DDPM"
    done
}

exp_4() { exp_4_train; exp_4_eval; }

exp_5_train() {
    for dataset in $SICAP_DATASETS; do
        should_run vae  && run_loop --skip-eval "mlp/vae/sicap"  "$dataset" "$SEEDS" "$LRS_VAE"
        should_run ddpm && run_loop --skip-eval "mlp/ddpm/sicap" "$dataset" "$SEEDS" "$LRS_DDPM"
    done
}

exp_5_eval() {
    for dataset in $SICAP_DATASETS; do
        should_run vae  && run_loop --skip-train "mlp/vae/sicap"  "$dataset" "$SEEDS" "$LRS_VAE"
        should_run ddpm && run_loop --skip-train "mlp/ddpm/sicap" "$dataset" "$SEEDS" "$LRS_DDPM"
    done
}

exp_5() { exp_5_train; exp_5_eval; }

exp_6() {
    for dataset in $SICAP_DATASETS; do
        for t in $T_VALUES; do
            run_timed "mlp_ddpm_sicap/$dataset/T$t" \
                --mode reconstruction-method \
                --experiment "mlp/ddpm/sicap" --dataset "$dataset" \
                --lr "$BEST_LR_DDPM" --seed "$BEST_SEED" \
                --n-score-steps "$t" --skip-train
        done
    done
    for dataset in $PATH_DATASETS; do
        for t in $T_VALUES; do
            run_timed "unet_ddpm_path/$dataset/T$t" \
                --mode reconstruction-method \
                --experiment "unet/ddpm/base" --dataset "$dataset" \
                --lr "$LRS_UNET_DDPM" --seed "$SEEDS_UNET" \
                --n-score-steps "$t" --skip-train \
                --skip-scores recon_single recon_multi
        done
    done
}

summary() {
    run_python --mode stats-summary --logs-dir "$LOGS_DIR" --out-csv "$OUT_CSV"
    log_ok "summary saved to $OUT_CSV"
}

exp_all() {
    exp_0; exp_1; exp_2; exp_3; exp_4; exp_5; exp_6; summary
}

exp_all_eval() {
    exp_0; exp_1_eval; exp_2_eval; exp_3_eval; exp_4_eval; exp_5_eval; exp_6; summary
}

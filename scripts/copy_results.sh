#!/bin/bash

S=42; T=10
LR_TOY=0.001
LR_VAE_M=0.0001;    LR_DDPM_M=0.0001
LR_VAE_UM=0.0001;   LR_DDPM_UM=0.0001
LR_VAE_P=0.0001;    LR_DDPM_P=0.0001
LR_VAE_UP=0.0001;   LR_DDPM_UP=0.0001
LR_VAE_SC1=0.0001;  LR_DDPM_SC1=0.0001
LR_VAE_SC12=0.0005; LR_DDPM_SC12=0.001

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/figures"

cp_fig() {
    local src="$ROOT/$1" dst="$DEST/$2"
    [[ -f "$src" ]] \
        && { mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; } \
        || echo "NOT FOUND: $1"
}

copy_vae() {
    local id="$1" dst="$2"
    local p="results/evaluation/$id/plots"
    cp_fig "$p/reconstructions_compact.pdf" "$dst/recon_compact.pdf"
    cp_fig "$p/reconstructions_full.pdf"    "$dst/recon_full.pdf"
    cp_fig "$p/embedding_vs_ood.pdf"        "$dst/embedding.pdf"
    for sc in elbo recon latent_knn latent_mah; do
        cp_fig "$p/ood_scores_$sc.pdf" "$dst/ood_$sc.pdf"
    done
}

copy_ddpm() {
    local id="$1" dst="$2"
    local p="results/evaluation/$id/plots"
    cp_fig "$p/ddpm_timestep_grid_compact.pdf"        "$dst/grid_compact.pdf"
    cp_fig "$p/ddpm_timestep_grid_full.pdf"           "$dst/grid_full.pdf"
    cp_fig "$p/ddpm_denoising_trajectory_compact.pdf" "$dst/traj_compact.pdf"
    cp_fig "$p/ddpm_denoising_trajectory_full.pdf"    "$dst/traj_full.pdf"
    for sc in noise_single noise_multi_mse noise_multi_cosine residual_mah residual_knn; do
        cp_fig "$p/ood_scores_$sc.pdf" "$dst/ood_$sc.pdf"
    done
}

for ds in blobs moons; do
    cp_fig "results/evaluation/mlp_vae_${ds}_s${S}_lr${LR_TOY}/plots/reconstructions.pdf"        "toy/$ds/vae_recon.pdf"
    cp_fig "results/evaluation/mlp_ddpm_${ds}_s${S}_lr${LR_TOY}_t${T}/plots/reconstructions.pdf" "toy/$ds/ddpm_recon.pdf"
done

copy_vae  "mlp_vae_mnist_s${S}_lr${LR_VAE_M}"           "mnist/mlp"
copy_ddpm "mlp_ddpm_mnist_s${S}_lr${LR_DDPM_M}_t${T}"   "mnist/mlp"
copy_vae  "unet_vae_mnist_s${S}_lr${LR_VAE_UM}"         "mnist/unet"
copy_ddpm "unet_ddpm_mnist_s${S}_lr${LR_DDPM_UM}_t${T}" "mnist/unet"

copy_vae  "mlp_vae_pathmnist_c1_s${S}_lr${LR_VAE_P}"           "pathmnist/mlp"
copy_ddpm "mlp_ddpm_pathmnist_c1_s${S}_lr${LR_DDPM_P}_t${T}"   "pathmnist/mlp"

for ds in pathmnist_c1 pathmnist_c2; do
    tag="${ds#pathmnist_}"
    copy_vae  "unet_vae_${ds}_s${S}_lr${LR_VAE_UP}"          "pathmnist/unet/$tag"
    copy_ddpm "unet_ddpm_${ds}_s${S}_lr${LR_DDPM_UP}_t${T}"  "pathmnist/unet/$tag"
done

for ds in sicap_c1 sicap_c12; do
    tag="${ds#sicap_}"
    lr_v="LR_VAE_SC${tag^^}"; lr_d="LR_DDPM_SC${tag^^}"
    p_vae="results/evaluation/mlp_vae_${ds}_s${S}_lr${!lr_v}/plots"
    p_ddpm="results/evaluation/mlp_ddpm_${ds}_s${S}_lr${!lr_d}_t${T}/plots"
    cp_fig "$p_vae/embedding_vs_ood.pdf" "sicap/$tag/vae_embedding.pdf"
    for sc in elbo recon latent_knn latent_mah; do
        cp_fig "$p_vae/ood_scores_$sc.pdf"  "sicap/$tag/vae_ood_$sc.pdf"
    done
    for sc in noise_single noise_multi_mse noise_multi_cosine residual_mah residual_knn; do
        cp_fig "$p_ddpm/ood_scores_$sc.pdf" "sicap/$tag/ddpm_ood_$sc.pdf"
    done
done

SUM="results/summary"
cp_fig "$SUM/comparison.csv"                   "tables/comparison.csv"
cp_fig "$SUM/ablation_t_auroc_sicap.pdf"       "tables/ablation_t_sicap.pdf"
cp_fig "$SUM/ablation_t_auroc_pathmnist.pdf"   "tables/ablation_t_pathmnist.pdf"
cp_fig "$SUM/sicap_methods_comparison.pdf"     "tables/sicap_comparison.pdf"
cp_fig "$SUM/pathmnist_methods_comparison.pdf" "tables/pathmnist_comparison.pdf"

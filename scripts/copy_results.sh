#!/bin/bash
# Copia figuras y tablas referenciadas en cap3, cap4 y el apéndice a figures/ y tables/.
# Excluye training curves: éstas se descargan directamente desde wandb.
# Convención: results/a/b/c.pdf → figures/results-a-b-c.pdf  (coincide con \includegraphics{} del tex)

set -euo pipefail

S=42; T=10
LR_TOY=0.001
LR_VAE_M=0.0001;    LR_DDPM_M=0.0001
LR_VAE_P=0.0001;    LR_DDPM_P=0.0001
LR_VAE_SC1=0.0001;  LR_DDPM_SC1=0.0001
LR_VAE_SC12=0.0005; LR_DDPM_SC12=0.001
LR_ISS=0.0001  # input_space_vs_ood (sin prefijo mlp_)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIGS="$ROOT/figures"
TABS="$ROOT/tables"
mkdir -p "$FIGS" "$TABS"

# Copia results/<path>.pdf → figures/results-<path-con-guiones>.pdf
cp_fig() {
    local rel="$1"
    local src="$ROOT/$rel"
    local dst="$FIGS/$(echo "$rel" | tr '/' '-')"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
    else
        echo "NOT FOUND: $rel"
    fi
}

# ── Capítulo 3: espacio latente vs OOD ───────────────────────────────────────
for ds in moons blobs; do
    cp_fig "results/training/mlp_vae_${ds}_s${S}_lr${LR_TOY}/plots/input_space_vs_ood.pdf"
done
cp_fig "results/training/mlp_vae_mnist_s${S}_lr${LR_VAE_M}/plots/input_space_vs_ood.pdf"
cp_fig "results/training/mlp_vae_pathmnist_c1_s${S}_lr${LR_VAE_P}/plots/input_space_vs_ood.pdf"
cp_fig "results/training/mlp_vae_sicap_c12_s${S}_lr${LR_ISS}/plots/input_space_vs_ood.pdf"

# ── Capítulo 4 · §4.1.1  Sintéticos ─────────────────────────────────────────
for ds in moons blobs; do
    cp_fig "results/evaluation/mlp_vae_${ds}_s${S}_lr${LR_TOY}/plots/reconstructions.pdf"
done

# ── Apéndice B: scores OOD de MNIST y PathMNIST ─────────────────────────────
copy_vae_scores() {
    local id="$1"
    local p="results/evaluation/$id/plots"
    for sc in recon elbo latent_knn latent_mah; do
        cp_fig "$p/ood_scores_$sc.pdf"
    done
}

copy_ddpm_scores() {
    local id="$1"
    local p="results/evaluation/$id/plots"
    for sc in noise_single noise_multi_mse noise_multi_cosine \
              recon_single recon_multi residual_mah residual_knn; do
        cp_fig "$p/ood_scores_$sc.pdf"
    done
}

copy_vae_scores  "mlp_vae_mnist_s${S}_lr${LR_VAE_M}"
copy_ddpm_scores "mlp_ddpm_mnist_s${S}_lr${LR_DDPM_M}_t${T}"
copy_vae_scores  "mlp_vae_pathmnist_c1_s${S}_lr${LR_VAE_P}"
copy_ddpm_scores "mlp_ddpm_pathmnist_c1_s${S}_lr${LR_DDPM_P}_t${T}"

# ── Capítulo 4 · §4.1.2  VAE MNIST / PathMNIST ──────────────────────────────
copy_vae_eval() {
    local id="$1"
    local p="results/evaluation/$id/plots"
    cp_fig "$p/reconstructions_compact.pdf"
    cp_fig "$p/embedding_vs_ood.pdf"
}

copy_vae_eval "mlp_vae_mnist_s${S}_lr${LR_VAE_M}"
copy_vae_eval "mlp_vae_pathmnist_c1_s${S}_lr${LR_VAE_P}"

# ── Capítulo 4 · §4.1.3  DDPM MNIST / PathMNIST ─────────────────────────────
copy_ddpm_eval() {
    local id="$1"
    local p="results/evaluation/$id/plots"
    cp_fig "$p/ddpm_timestep_grid_compact.pdf"
    cp_fig "$p/ddpm_denoising_trajectory_compact.pdf"
}

copy_ddpm_eval "mlp_ddpm_mnist_s${S}_lr${LR_DDPM_M}_t${T}"
copy_ddpm_eval "mlp_ddpm_pathmnist_c1_s${S}_lr${LR_DDPM_P}_t${T}"

# ── Capítulo 4 · §4.1.4  SICAP (scores VAE + DDPM) ──────────────────────────
copy_sicap_vae() {
    local id="$1"
    local p="results/evaluation/$id/plots"
    for sc in recon elbo latent_knn latent_mah; do
        cp_fig "$p/ood_scores_$sc.pdf"
    done
}

copy_sicap_ddpm() {
    local id="$1"
    local p="results/evaluation/$id/plots"
    for sc in noise_single noise_multi_mse noise_multi_cosine residual_mah residual_knn; do
        cp_fig "$p/ood_scores_$sc.pdf"
    done
}

copy_sicap_vae  "mlp_vae_sicap_c1_s${S}_lr${LR_VAE_SC1}"
copy_sicap_ddpm "mlp_ddpm_sicap_c1_s${S}_lr${LR_DDPM_SC1}_t${T}"
copy_sicap_vae  "mlp_vae_sicap_c12_s${S}_lr${LR_VAE_SC12}"
copy_sicap_ddpm "mlp_ddpm_sicap_c12_s${S}_lr${LR_DDPM_SC12}_t${T}"

# ── Capítulo 4 · §4.2  Comparativa y ablación ────────────────────────────────
SUM="results/summary"
cp_fig "$SUM/sicap_methods_comparison.pdf"
cp_fig "$SUM/ablation_t_auroc_sicap.pdf"

# Tablas .tex → tables/ (referenciadas con \input{tables/...})
for f in \
    "table_comparison_sicap.tex" \
    "table_ablation_t_sicap.tex" \
    "table_results_sicap.tex" \
    "table_training_times.tex"; do
    src="$ROOT/$SUM/$f"
    if [[ -f "$src" ]]; then
        cp "$src" "$TABS/$f"
    else
        echo "NOT FOUND: $SUM/$f"
    fi
done

echo "Done. Figures → $FIGS   Tables → $TABS"

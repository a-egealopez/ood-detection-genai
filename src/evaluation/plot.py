from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from omegaconf import DictConfig
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

import wandb
from src.artifacts import save_figure
from src.viz.style import apply_paper_style

if TYPE_CHECKING:
    from src.evaluation.extract import ScoreBundle


# apply a consistent plotting style at import time
try:
    apply_paper_style(None)
except Exception:
    pass


def render_cell(ax: plt.Axes, tensor: torch.Tensor, is_image: bool, color: str) -> None:
    arr = tensor.squeeze().cpu().numpy()
    if is_image:
        if arr.size == 784:
            ax.imshow(arr.reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
        else:
            img = arr.reshape(3, 28, 28).transpose(1, 2, 0)
            ax.imshow(np.clip(img * 0.5 + 0.5, 0, 1))
    else:
        ax.plot(arr, color=color, lw=1.0)
        ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_score_distribution(
    id_scores: np.ndarray, ood_scores: np.ndarray, mode: str, auroc: float, ax: plt.Axes,
    threshold: float | None = None,
) -> None:
    bins = np.linspace(
        min(id_scores.min(), ood_scores.min()), max(id_scores.max(), ood_scores.max()), 60
    )
    ax.hist(id_scores, bins=bins, density=True, alpha=0.45, color="steelblue", label="ID")
    ax.hist(ood_scores, bins=bins, density=True, alpha=0.45, color="tomato", label="OOD")
    for s, c in [(id_scores, "steelblue"), (ood_scores, "tomato")]:
        if s.std() > 0:
            kde = gaussian_kde(s, bw_method=0.3)
            xs = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=c, linewidth=2)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold = {threshold:.3f}")
    ax.set_title("Score Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("OOD Score (higher = more OOD)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_roc_curve(
    id_scores: np.ndarray, ood_scores: np.ndarray, mode: str, auroc: float, ax: plt.Axes
) -> None:
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([id_scores, ood_scores]))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"ROC Curve", fontsize=11, fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(True, alpha=0.3)


_SCORE_GROUPS: list[list[str]] = [
    ["recon", "elbo"],
    ["latent_knn", "latent_mah"],
    ["noise_single", "noise_multi_mse", "noise_multi_cosine"],
    ["residual_mah", "residual_knn"],
]


def plot_ood_evaluation(
    scores: ScoreBundle,
    aurocs: dict[str, float],
    active: list[tuple[str, str, str]],
    run=None,
    plots_dir: Path | None = None,
    cfg: DictConfig | None = None,
    thresholds: dict[str, dict] | None = None,
    labels_map: dict[str, str] | None = None,
) -> None:

    active_set = {m for m, _, _ in active}
    active_map = {m: (id_a, ood_a) for m, id_a, ood_a in active}

    # determine target width from cfg
    textwidth = 6.0
    try:
        if cfg is not None:
            textwidth = float(cfg.viz.get("textwidth_in", textwidth))
    except Exception:
        pass

    for group in _SCORE_GROUPS:
        chunk = [(m, *active_map[m]) for m in group if m in active_set]
        if not chunk:
            continue

        # render one figure per active mode for clarity and publication sizing
        for mode, id_attr, ood_attr in chunk:

            threshold_val = thresholds[mode]["threshold"] if thresholds and mode in thresholds else None
            fig, axes = plt.subplots(1, 2, figsize=(textwidth, 3.5))

            pretty_name = labels_map.get(mode, mode) if labels_map else mode
            fig.suptitle(f"OOD Score Evaluation — {pretty_name}", fontsize=14, fontweight="bold")

            _plot_score_distribution(
                getattr(scores, id_attr), getattr(scores, ood_attr),
                pretty_name, aurocs.get(mode, float("nan")), axes[0],
                threshold=threshold_val,
            )
            _plot_roc_curve(
                getattr(scores, id_attr), getattr(scores, ood_attr),
                pretty_name, aurocs.get(mode, float("nan")), axes[1],
            )

            plt.tight_layout()
            mode_name = mode
            if plots_dir is not None:
                save_figure(
                    fig=fig,
                    out_path=plots_dir / f"ood_scores_{mode_name}.png",
                    run=run,
                    image_key=f"eval/ood_scores/{mode_name}",
                    artifact_type="plot",
                    artifact_prefix="ood-scores",
                    metadata={"modes": mode_name},
                    png_dpi=int(cfg.viz.get("savefig_dpi", 300)) if cfg is not None else 300,
                )
            elif run is not None:
                run.log({f"eval/ood_scores/{mode_name}": wandb.Image(fig)})
            plt.close(fig)


def _standardize_scale(zs: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(zs)


def _pca_preprocess(
    zs: np.ndarray, cfg: DictConfig, components_key: str | None = None
) -> np.ndarray:
    n = int(
        cfg.viz.get(components_key, cfg.viz.pca_n_components)
        if components_key
        else cfg.viz.pca_n_components
    )
    return zs if zs.shape[1] <= n else PCA(n_components=n, random_state=cfg.seed).fit_transform(zs)


def project_pca(zs: np.ndarray, cfg: DictConfig) -> np.ndarray:
    return PCA(n_components=2, random_state=cfg.seed).fit_transform(_standardize_scale(zs))


def _project_tsne(zs: np.ndarray, cfg: DictConfig) -> np.ndarray:
    zs_r = _pca_preprocess(_standardize_scale(zs), cfg)
    return TSNE(
        n_components=2,
        perplexity=max(1, min(cfg.viz.tsne_perplexity, zs_r.shape[0] - 1)),
        random_state=cfg.seed,
        max_iter=1000,
        verbose=0,
    ).fit_transform(zs_r)


def _project_umap(zs: np.ndarray, cfg: DictConfig) -> np.ndarray:
    import warnings
    zs_r = _pca_preprocess(_standardize_scale(zs), cfg, components_key="umap_pca_init_components")
    n_neighbors = min(int(cfg.viz.get("umap_n_neighbors", 15)), zs_r.shape[0] - 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=float(cfg.viz.get("umap_min_dist", 0.1)),
            metric=str(cfg.viz.get("umap_metric", "euclidean")),
            random_state=cfg.seed,
            verbose=False,
        ).fit_transform(zs_r)


PROJECTOR_REGISTRY: dict[str, callable] = {
    "pca": project_pca,
    "tsne": _project_tsne,
    "umap": _project_umap,
}
DEFAULT_PROJECTORS: list[str] = ["pca", "umap"]


def _scatter_panel(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    ax: plt.Axes,
    label_map: dict | None = None,
    ood_label: int | None = None,
) -> None:
    cmap = plt.colormaps["tab10"]
    id_labels = [lbl for lbl in np.unique(labels) if lbl != ood_label]
    for idx, lbl in enumerate(np.unique(labels)):
        if lbl == ood_label:
            continue
        name = label_map.get(lbl, str(lbl)) if label_map else str(lbl)
        ax.scatter(
            coords[labels == lbl, 0],
            coords[labels == lbl, 1],
            c=[cmap(idx / max(len(id_labels) - 1, 1))],
            label=name,
            alpha=0.5,
            s=8,
            linewidths=0,
        )
    if ood_label is not None:
        name = label_map.get(ood_label, "OOD") if label_map else "OOD"
        ax.scatter(
            coords[labels == ood_label, 0],
            coords[labels == ood_label, 1],
            c="red",
            label=name,
            alpha=0.7,
            s=20,
            marker="x",
            linewidths=0.8,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(markerscale=2, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_embeddings(
    zs: np.ndarray,
    labels: np.ndarray,
    cfg: DictConfig,
    title: str,
    projectors: list[str] | None = None,
    label_map: dict | None = None,
    ood_label: int | None = None,
    save_name: str | None = None,
    run=None,
    wandb_key: str | None = None,
) -> plt.Figure:
    """Render one scatter panel per projector for a set of latent embeddings."""

    projectors = projectors or list(cfg.viz.get("projectors", DEFAULT_PROJECTORS))
    unknown = [p for p in projectors if p not in PROJECTOR_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown projector(s): {unknown}. Available: {list(PROJECTOR_REGISTRY)}")

    # use configured textwidth and compact height for publication figures
    textwidth = 6.0
    try:
        textwidth = float(cfg.viz.get("textwidth_in", textwidth))
    except Exception:
        pass
    width = textwidth * len(projectors)
    fig, axes = plt.subplots(1, len(projectors), figsize=(width, 3.5))
    if len(projectors) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for ax, name in zip(axes, projectors, strict=False):
        _scatter_panel(
            PROJECTOR_REGISTRY[name](zs, cfg),
            labels,
            f"{name.upper()} (2D)",
            ax,
            label_map,
            ood_label,
        )
    plt.tight_layout()

    if save_name:
        save_figure(
            fig=fig,
            out_path=Path(cfg.viz.plots_dir) / f"{save_name}.png",
            run=run if wandb_key else None,
            image_key=wandb_key,
            artifact_type="plot",
            artifact_prefix="embedding",
            metadata={
                "save_name": save_name,
                "projectors": list(projectors),
                "dataset": str(cfg.data.dataset),
            },
        )
    elif run is not None and wandb_key:
        run.log({wandb_key: wandb.Image(fig)})

    return fig

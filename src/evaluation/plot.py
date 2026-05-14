from pathlib import Path

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
from src.evaluation.extract import ScoreBundle


def render_cell(ax: plt.Axes, tensor: torch.Tensor, is_image: bool, color: str) -> None:
    arr = tensor.squeeze().cpu().numpy()
    if is_image:
        ax.imshow(arr.reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
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
    id_scores: np.ndarray, ood_scores: np.ndarray, mode: str, auroc: float, ax: plt.Axes
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
    ax.set_title(f"Score Distribution [{mode}]  AUROC={auroc:.4f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("OOD Score (higher = more OOD)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_roc_curve(
    id_scores: np.ndarray, ood_scores: np.ndarray, mode: str, auroc: float, ax: plt.Axes
) -> None:
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([id_scores, ood_scores]))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC [{mode}]  AUROC={auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"ROC Curve [{mode}]", fontsize=11, fontweight="bold")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_per_class_boxplot(
    id_scores: np.ndarray, id_labels: np.ndarray, ood_scores: np.ndarray, mode: str
) -> plt.Figure:
    classes = sorted(np.unique(id_labels).tolist())
    id_mu, id_std = id_scores.mean(), id_scores.std()
    ood_mu, ood_std = ood_scores.mean(), ood_scores.std()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        [id_scores[id_labels == c] for c in classes],
        labels=[str(c) for c in classes],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="navy", linewidth=2),
        showfliers=False,
    )
    ax.axhline(id_mu, color="green", lw=1.5, ls="-.", label=f"ID mean = {id_mu:.5f}")
    ax.axhspan(id_mu - id_std, id_mu + id_std, color="green", alpha=0.10, label="ID ±1σ")
    ax.axhline(ood_mu, color="tomato", lw=2, ls="--", label=f"OOD mean = {ood_mu:.5f}")
    ax.axhspan(ood_mu - ood_std, ood_mu + ood_std, color="tomato", alpha=0.15, label="OOD ±1σ")
    ax.set_title(f"Per-Class Score [{mode}]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ood_evaluation(
    scores: ScoreBundle,
    id_labels: np.ndarray,
    aurocs: dict[str, float],
    active: list[tuple[str, str, str]],
    run=None,
    boxplot_mode: str = "recon",
) -> None:
    """Render score distribution, ROC, and per-class boxplot figures."""

    for i in range(0, len(active), 2):
        chunk = active[i : i + 2]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(16, 5 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[np.newaxis, :]
        mode_names = "+".join(m for m, _, _ in chunk)
        fig.suptitle(f"OOD Evaluation — {mode_names}", fontsize=14, fontweight="bold")
        for row, (mode, id_attr, ood_attr) in enumerate(chunk):
            _plot_score_distribution(
                getattr(scores, id_attr),
                getattr(scores, ood_attr),
                mode,
                aurocs[mode],
                axes[row, 0],
            )
            _plot_roc_curve(
                getattr(scores, id_attr),
                getattr(scores, ood_attr),
                mode,
                aurocs[mode],
                axes[row, 1],
            )
        plt.tight_layout()
        plt.show()
        if run is not None:
            run.log({f"eval/plots/ood/{mode_names}": wandb.Image(fig)})
        plt.close(fig)

    id_scores = getattr(scores, f"id_{boxplot_mode}", np.array([]))
    ood_scores = getattr(scores, f"ood_{boxplot_mode}", np.array([]))
    if id_scores.size > 0 and ood_scores.size > 0:
        fig_cls = _plot_per_class_boxplot(id_scores, id_labels, ood_scores, mode=boxplot_mode)
        plt.show()
        if run is not None:
            run.log({"eval/plots/per_class_scores": wandb.Image(fig_cls)})
        plt.close(fig_cls)
    else:
        print(f"  Skipping per-class boxplot: '{boxplot_mode}' has no scores.")


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
    zs_r = _pca_preprocess(_standardize_scale(zs), cfg, components_key="umap_pca_init_components")
    n_neighbors = min(int(cfg.viz.get("umap_n_neighbors", 15)), zs_r.shape[0] - 1)
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(cfg.viz.get("umap_min_dist", 0.1)),
        metric=str(cfg.viz.get("umap_metric", "euclidean")),
        n_jobs=-1,
        random_state=cfg.seed,
        verbose=False,
    ).fit_transform(zs_r)


PROJECTOR_REGISTRY: dict[str, callable] = {
    "pca": project_pca,
    "tsne": _project_tsne,
    "umap": _project_umap,
}
DEFAULT_PROJECTORS: list[str] = ["pca", "tsne", "umap"]


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

    fig, axes = plt.subplots(1, len(projectors), figsize=(8 * len(projectors), 6))
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

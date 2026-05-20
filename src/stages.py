from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.evaluation as eval
import src.training as train
from src.artifacts import save_figure
from src.evaluation.plot import plot_ood_evaluation, project_pca, render_cell

_IMAGE_FLAT_DIM = 784


@dataclass(frozen=True)
class StageContext:
    model: torch.nn.Module
    loaders: dict
    cfg: object
    device: torch.device
    run: object
    experiment_id: str


@dataclass(frozen=True)
class DatasetMetadata:
    id_name: str
    ood_name: str
    ood_label: int

    @property
    def label_map(self) -> dict[int, str]:
        return {
            **{i: f"{self.id_name} {i}" for i in range(self.ood_label)},
            self.ood_label: f"{self.ood_name} (OOD)",
        }

    @classmethod
    def from_cfg(cls, cfg) -> DatasetMetadata:
        return cls(
            id_name=cfg.data.get("id_name", "ID"),
            ood_name=cfg.data.get("ood_name", "OOD"),
            ood_label=int(cfg.data.ood_label),
        )


def stage_input_space_viz(ctx: StageContext) -> None:
    metadata = DatasetMetadata.from_cfg(ctx.cfg)

    id_vecs, id_labels = _collect_raw_vectors(ctx.loaders["id_eval"])
    ood_vecs, _ = _collect_raw_vectors(ctx.loaders["ood_eval"])
    combined_vecs = np.concatenate([id_vecs, ood_vecs])
    combined_labels = np.concatenate([id_labels, np.full(len(ood_vecs), metadata.ood_label)])

    _show_save_log(
        eval.plot_embeddings(
            id_vecs,
            id_labels,
            ctx.cfg,
            title=f"Input Space — {metadata.id_name} (ID)",
            label_map={i: str(i) for i in range(metadata.ood_label)},
            ood_label=None,
            run=ctx.run,
            wandb_key="viz/input_space",
        ),
        ctx,
        "input_space.png",
        "viz/input_space",
    )
    _show_save_log(
        eval.plot_embeddings(
            combined_vecs,
            combined_labels,
            ctx.cfg,
            title=f"Input Space — {metadata.id_name} (ID) vs {metadata.ood_name} (OOD)",
            label_map=metadata.label_map,
            ood_label=metadata.ood_label,
            run=ctx.run,
            wandb_key="viz/input_space_vs_ood",
        ),
        ctx,
        "input_space_vs_ood.png",
        "viz/input_space_vs_ood",
    )


def stage_training(ctx: StageContext) -> list[dict]:
    history = train.train_model(ctx.model, ctx.loaders, ctx.cfg, ctx.device, ctx.run)
    loss_keys = list(history[0].keys())
    epochs = range(1, len(history) + 1)
    colors = ["steelblue", "darkorange", "mediumseagreen", "tomato", "orchid"]

    fig, axes = plt.subplots(1, len(loss_keys), figsize=(6 * len(loss_keys), 4))
    fig.suptitle("Training Loss History", fontsize=14, fontweight="bold")

    for ax, key, color in zip(np.atleast_1d(axes), loss_keys, colors, strict=False):
        vals = [h[key] for h in history]
        ax.plot(epochs, vals, color=color, lw=2)
        ax.fill_between(epochs, vals, color=color, alpha=0.15)
        ax.set_title(key.capitalize(), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _show_save_log(fig, ctx, "training_curves.png", "train/curves")
    ctx.run.log({"train/status": "complete"})
    return history


def stage_reconstruction_viz_toy(ctx: StageContext) -> None:
    metadata = DatasetMetadata.from_cfg(ctx.cfg)
    _show_save_log(
        _build_toy_recon_fig(ctx, metadata),
        ctx,
        "reconstructions.png",
        "eval/reconstructions",
    )


def stage_reconstruction_viz_vae(ctx: StageContext) -> None:
    metadata = DatasetMetadata.from_cfg(ctx.cfg)
    _show_save_log(
        _build_reconstruction_fig_vae(ctx, metadata),
        ctx,
        "reconstructions.png",
        "eval/reconstructions",
    )


def stage_reconstruction_viz_ddpm(ctx: StageContext) -> None:
    metadata = DatasetMetadata.from_cfg(ctx.cfg)
    _show_save_log(
        _build_ddpm_timestep_grid(ctx, metadata),
        ctx,
        "ddpm_timestep_grid.png",
        "eval/ddpm_timestep_grid",
    )
    _show_save_log(
        _build_ddpm_denoising_trajectory(ctx, metadata),
        ctx,
        "ddpm_denoising_trajectory.png",
        "eval/ddpm_denoising_trajectory",
    )


def stage_evaluation(ctx: StageContext):
    metadata = DatasetMetadata.from_cfg(ctx.cfg)
    embed_label = "Embedding Space"

    scores = eval.compute_all_scores(ctx.model, ctx.loaders, ctx.device, ctx.cfg)
    if ctx.cfg.wandb.enabled:
        ctx.run.log(_build_scores_payload(scores))

    results = eval.evaluate_ood(
        scores,
        ctx.cfg,
        ctx.run,
        experiment_id=ctx.experiment_id,
    )

    embeddings_id, labels_id = eval.extract_representations(
        ctx.model, ctx.loaders["id_eval"], ctx.device
    )
    embeddings_ood, _ = eval.extract_representations(ctx.model, ctx.loaders["ood_eval"], ctx.device)

    if bool(ctx.cfg.viz.get("plot_embeddings", True)):
        eval.plot_embeddings(
            np.concatenate([embeddings_id, embeddings_ood]),
            np.concatenate([labels_id, np.full(len(embeddings_ood), metadata.ood_label)]),
            ctx.cfg,
            title=f"Final {embed_label} — {metadata.id_name} (ID) vs {metadata.ood_name} (OOD)",
            label_map=metadata.label_map,
            ood_label=metadata.ood_label,
            run=ctx.run,
            wandb_key="eval/embedding_vs_ood",
            save_name="embedding_vs_ood",
        )
        plt.show()

    raw_vecs, raw_labs, recon_vecs, recon_labs = _collect_recon_vectors(ctx)
    _plot_embedding_panel(
        ctx, raw_vecs, raw_labs, embeddings_id, labels_id, recon_vecs, recon_labs, embed_label
    )

    from src.evaluation.evaluate import _active_modes
    active = _active_modes(scores)
    aurocs = {mode: results.metrics[mode]["auroc"] for mode in results.metrics}
    plots_dir = Path(ctx.cfg.viz.plots_dir)
    plot_ood_evaluation(scores, aurocs, active, run=ctx.run, plots_dir=plots_dir)

    ctx.run.log({"eval/status": "complete"})
    return results


def print_summary(cfg, results) -> None:
    model_type = cfg.model.model_type.upper()
    print("\n" + "=" * 65)
    print(f"          {model_type} OOD DETECTION — FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Mode':<12}  {'AUROC':>8}  {'AUPR':>8}  {'FPR@95':>8}")
    print("  " + "-" * 42)
    for mode, m in results.metrics.items():
        print(f"  {mode:<12}  {m['auroc']:>8.4f}  {m['aupr']:>8.4f}  {m['fpr_at_95_tpr']:>8.4f}")
    print("-" * 65)
    print(json.dumps(results.to_dict(), indent=4))
    print("=" * 65)


def _build_toy_recon_fig(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
    """2D scatter: original vs reconstructed for ID and OOD toy data."""
    ctx.model.eval()
    with torch.no_grad():
        x_id = torch.cat([x for x, _ in ctx.loaders["id_eval"]], dim=0).to(ctx.device)
        x_ood = torch.cat([x for x, _ in ctx.loaders["ood_eval"]], dim=0).to(ctx.device)
        x_recon_id = ctx.model(x_id)[0]
        x_recon_ood = ctx.model(x_ood)[0]

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.view(t.shape[0], -1).cpu().numpy()

    id_np, ood_np = to_np(x_id), to_np(x_ood)
    recon_id_np, recon_ood_np = to_np(x_recon_id), to_np(x_recon_ood)

    model_type = ctx.cfg.model.model_type.upper()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"[{model_type}] Original vs Reconstructed — "
        f"{metadata.id_name} (ID) vs {metadata.ood_name} (OOD)",
        fontsize=13, fontweight="bold",
    )
    kw = dict(s=14, alpha=0.7)
    axes[0].scatter(id_np[:, 0], id_np[:, 1], c="steelblue", label=f"{metadata.id_name} (ID)", **kw)
    axes[0].scatter(ood_np[:, 0], ood_np[:, 1], c="tomato", label=f"{metadata.ood_name} (OOD)", **kw)
    axes[0].set_title("Input space", fontweight="bold")
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(recon_id_np[:, 0], recon_id_np[:, 1], c="steelblue",
                    label=f"Recon {metadata.id_name} (ID)", **kw)
    axes[1].scatter(recon_ood_np[:, 0], recon_ood_np[:, 1], c="darkorange",
                    label=f"Recon {metadata.ood_name} (OOD)", **kw)
    axes[1].set_title("Reconstructed space", fontweight="bold")
    axes[1].legend(markerscale=2, fontsize=9)
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    ctx.model.train()
    return fig


def _build_reconstruction_fig_vae(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
    ctx.model.eval()
    x_id = next(iter(ctx.loaders["id_eval"]))[0][:8].to(ctx.device)
    x_ood = next(iter(ctx.loaders["ood_eval"]))[0][:8].to(ctx.device)
    with torch.no_grad():
        x_recon_id, _, _ = ctx.model(x_id)
        x_recon_ood, _, _ = ctx.model(x_ood)

    is_image = int(ctx.cfg.data.input_dim) == _IMAGE_FLAT_DIM and bool(
        ctx.cfg.data.get("is_image", True)
    )
    model_type = ctx.cfg.model.model_type.upper()

    fig, axes = plt.subplots(4, 8, figsize=(16, 9))
    fig.suptitle(
        f"[{model_type}] Original vs Reconstructed — {metadata.id_name} (ID) vs {metadata.ood_name} (OOD)",
        fontsize=13,
        fontweight="bold",
    )

    for row, (title, batch, color) in enumerate(
        [
            (f"{metadata.id_name} original", x_id, "steelblue"),
            (f"{metadata.id_name} reconstructed", x_recon_id, "steelblue"),
            (f"{metadata.ood_name} original", x_ood, "tomato"),
            (f"{metadata.ood_name} reconstructed", x_recon_ood, "tomato"),
        ]
    ):
        axes[row, 0].set_ylabel(title, fontsize=9, fontweight="bold", color=color)
        for col in range(8):
            render_cell(axes[row, col], batch[col].cpu(), is_image, color)

    fig.text(
        0.99, 0.75, "ID", fontsize=14, fontweight="bold", color="steelblue", va="center", ha="right"
    )
    fig.text(
        0.99, 0.28, "OOD", fontsize=14, fontweight="bold", color="tomato", va="center", ha="right"
    )
    plt.tight_layout()
    return fig


def _build_ddpm_timestep_grid(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
    ctx.model.eval()

    max_t = int(ctx.model.num_train_timesteps) - 1
    keyframes = _ddpm_keyframes(max_t)

    x_id = next(iter(ctx.loaders["id_eval"]))[0][: len(keyframes)].to(ctx.device)
    x_ood = next(iter(ctx.loaders["ood_eval"]))[0][: len(keyframes)].to(ctx.device)

    t_values = torch.tensor(keyframes, device=ctx.device, dtype=torch.long)[: x_id.shape[0]]

    with torch.no_grad():
        x_recon_id, x_noisy_id, _ = ctx.model.reconstruct_at_t(
            x_id, t_values, noise=torch.randn_like(x_id)
        )
        x_recon_ood, x_noisy_ood, _ = ctx.model.reconstruct_at_t(
            x_ood, t_values, noise=torch.randn_like(x_ood)
        )

    is_image = (
        bool(ctx.cfg.data.get("is_image", False)) and int(ctx.cfg.data.input_dim) == _IMAGE_FLAT_DIM
    )
    model_type = ctx.cfg.model.model_type.upper()
    n_cols = len(keyframes) + 1

    fig, axes = plt.subplots(6, n_cols, figsize=(2.1 * n_cols, 12))
    fig.suptitle(
        f"[{model_type}] Diffusion Process over timesteps — {metadata.id_name} (ID) vs {metadata.ood_name} (OOD)",
        fontsize=15,
        fontweight="bold",
    )

    for row, (title, batch, color, t_vals) in enumerate(
        [
            (f"{metadata.id_name} Original", x_id, "steelblue", None),
            (f"{metadata.id_name} Noisy (x_t)", x_noisy_id, "steelblue", t_values),
            (f"{metadata.id_name} Reconstructed", x_recon_id, "steelblue", t_values),
            (f"{metadata.ood_name} Original", x_ood, "tomato", None),
            (f"{metadata.ood_name} Noisy (x_t)", x_noisy_ood, "tomato", t_values),
            (f"{metadata.ood_name} Reconstructed", x_recon_ood, "tomato", t_values),
        ]
    ):
        axes[row, 0].set_ylabel(title, fontsize=9, fontweight="bold", color=color)

        render_cell(axes[row, 0], batch[0].cpu(), is_image, color)

        for col in range(len(keyframes)):
            ax = axes[row, col + 1]
            render_cell(ax, batch[col].cpu(), is_image, color)
            if t_vals is not None:
                ax.set_title(f"t={t_vals[col].item()}", fontsize=7, color=color, pad=2)

    fig.text(
        0.99, 0.75, "ID", fontsize=16, fontweight="bold", color="steelblue", va="center", ha="right"
    )
    fig.text(
        0.99, 0.25, "OOD", fontsize=16, fontweight="bold", color="tomato", va="center", ha="right"
    )
    plt.tight_layout(rect=[0, 0, 0.98, 0.97])
    return fig


def _build_ddpm_denoising_trajectory(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
    ctx.model.eval()

    max_t = int(ctx.model.num_train_timesteps / 2)
    keyframes = _ddpm_keyframes(max_t)

    x_id = next(iter(ctx.loaders["id_eval"]))[0][: len(keyframes)].to(ctx.device)
    x_ood = next(iter(ctx.loaders["ood_eval"]))[0][: len(keyframes)].to(ctx.device)

    tensor_n_steps = torch.tensor(keyframes, device=ctx.device, dtype=torch.long)
    tensor_n_steps = tensor_n_steps[(tensor_n_steps >= 0) & (tensor_n_steps <= max_t)]
    n_steps = int(tensor_n_steps.shape[0])

    x_ref_id = x_id[:4]
    x_ref_ood = x_ood[:4]
    t_start = int(tensor_n_steps.max().item())
    captures = [int(t.item()) for t in tensor_n_steps]
    traj_id = ctx.model.denoise_trajectory(
        x_ref_id, t_start=t_start, capture_timesteps=captures, noise=torch.randn_like(x_ref_id)
    )
    traj_ood = ctx.model.denoise_trajectory(
        x_ref_ood, t_start=t_start, capture_timesteps=captures, noise=torch.randn_like(x_ref_ood)
    )

    is_image = (
        bool(ctx.cfg.data.get("is_image", False)) and int(ctx.cfg.data.input_dim) == _IMAGE_FLAT_DIM
    )
    model_type = ctx.cfg.model.model_type.upper()
    n_rows = x_ref_id.shape[0] + x_ref_ood.shape[0]

    fig, axes = plt.subplots(n_rows, n_steps + 1, figsize=(2.0 * (n_steps + 1), 1.9 * n_rows))
    fig.suptitle(
        f"[{model_type}] Real Reverse Denoising — ID vs OOD (high t -> low t)",
        fontsize=14,
        fontweight="bold",
    )

    for r in range(n_rows):
        is_ood_row = r >= x_ref_id.shape[0]
        local_r = r - x_ref_id.shape[0] if is_ood_row else r
        x_ref = x_ref_ood if is_ood_row else x_ref_id
        traj = traj_ood if is_ood_row else traj_id
        color = "tomato" if is_ood_row else "steelblue"
        tag = "OOD" if is_ood_row else "ID"

        render_cell(axes[r, 0], x_ref[local_r].cpu(), is_image, color)
        axes[r, 0].set_title(f"{tag} x (orig)", fontsize=8, color=color)

        for c, t in enumerate(reversed(tensor_n_steps.tolist()), start=1):
            token = f"x_t (t={t})" if t > 0 else "x_0 (final)"
            render_cell(axes[r, c], traj[int(t)][local_r].cpu(), is_image, color)
            axes[r, c].set_title(f"{tag} {token}", fontsize=8, color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _ddpm_keyframes(max_t: int) -> list[int]:
    t, frames = max_t, []
    while t >= 1:
        frames.append(t)
        t //= 2
    return frames[::-1]


def _show_save_log(fig, ctx: StageContext, filename: str, wandb_key: str) -> None:
    if ctx.cfg.viz.get("show_plots", False):
        plt.show()

    out_path = Path(ctx.cfg.viz.plots_dir) / filename

    save_figure(
        fig=fig,
        out_path=out_path,
        run=ctx.run if ctx.cfg.wandb.enabled else None,
        image_key=wandb_key,
        artifact_type="plot",
        artifact_prefix="plot",
        metadata={
            "experiment_name": str(ctx.cfg.experiment_name),
            "model_type": str(ctx.cfg.model.model_type),
            "dataset": str(ctx.cfg.data.dataset),
        },
    )
    plt.close(fig)


def _collect_raw_vectors(loader):
    vecs, labs = [], []
    for x, y in loader:
        vecs.append(x.view(x.size(0), -1).numpy())
        labs.append(y.numpy())
    return np.concatenate(vecs), np.concatenate(labs)


def _collect_recon_vectors(ctx: StageContext):
    raw_vecs, raw_labs, recon_vecs, recon_labs = [], [], [], []
    ctx.model.eval()
    with torch.no_grad():
        for x, y in ctx.loaders["id_eval"]:
            xr, _, _ = ctx.model(x.to(ctx.device))
            recon_vecs.append(xr.view(xr.size(0), -1).cpu().numpy())
            recon_labs.append(y.numpy())
            raw_vecs.append(x.view(x.size(0), -1).numpy())
            raw_labs.append(y.numpy())
    return (
        np.concatenate(raw_vecs),
        np.concatenate(raw_labs),
        np.concatenate(recon_vecs),
        np.concatenate(recon_labs),
    )


def _plot_embedding_panel(
    ctx: StageContext,
    raw_vecs,
    raw_labs,
    embeddings_id,
    labels_id,
    recon_vecs,
    recon_labs,
    embed_label,
) -> None:
    model_type = ctx.cfg.model.model_type.upper()
    cmap = plt.colormaps["tab10"].resampled(10)
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(
        f"[{model_type}] Input → {embed_label} → Reconstructed — ID",
        fontsize=13,
        fontweight="bold",
    )

    for panel, (vecs, labs, title) in enumerate(
        [
            (raw_vecs, raw_labs, "Input space (raw pixels)"),
            (embeddings_id, labels_id, embed_label),
            (recon_vecs, recon_labs, "Reconstructed space (x̂₀)"),
        ]
    ):
        coords = project_pca(vecs, ctx.cfg)
        for c in np.unique(labs).astype(int):
            m = labs == c
            axs[panel].scatter(
                coords[m, 0], coords[m, 1], c=[cmap(c)], s=8, alpha=0.5, label=str(c)
            )
        axs[panel].set_title(title, fontsize=10, fontweight="bold")
        axs[panel].axis("off")

    axs[0].legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    save_figure(
        fig=fig,
        out_path=Path(ctx.cfg.viz.plots_dir) / "embedding_panel.png",
        run=ctx.run if ctx.cfg.wandb.enabled else None,
        image_key="eval/embedding_panel",
        artifact_type="plot",
        artifact_prefix="panel",
        metadata={
            "experiment_name": str(ctx.cfg.experiment_name),
            "model_type": str(ctx.cfg.model.model_type),
            "dataset": str(ctx.cfg.data.dataset),
        },
    )
    plt.close(fig)


def _build_scores_payload(scores) -> dict:
    payload = {}
    all_modes = [
        "recon", "elbo", "knn", "mahalanobis",
        "noise_single", "noise_multi_mse", "noise_multi_cosine",
        "residual_mah", "residual_knn",
    ]
    for mode in all_modes:
        id_arr = getattr(scores, f"id_{mode}", np.array([]))
        ood_arr = getattr(scores, f"ood_{mode}", np.array([]))
        if id_arr.size > 0:
            payload[f"scores/id_{mode}_mean"] = float(id_arr.mean())
            payload[f"scores/ood_{mode}_mean"] = float(ood_arr.mean())
    return payload

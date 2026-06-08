from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.evaluation as eval
import src.training as train
import src.viz.style as style
from src.artifacts import save_figure
from src.evaluation.plot import plot_ood_evaluation, project_pca, render_cell


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
            ood_label=int(cfg.data.get("ood_label", 1)),
        )


MODE_LABELS: dict[str, str] = {
    "recon": "Reconstruction Error",
    "elbo": "ELBO",
    "latent_knn": "Latent k-NN",
    "latent_mah": "Latent Mahalanobis",
    "noise_single": "Single-step MSE",
    "noise_multi_mse": "Multi-step MSE (z-score)",
    "noise_multi_cosine": "Multi-step Cosine (z-score)",
    "residual_mah": "Residual Mahalanobis",
    "residual_knn": "Residual k-NN",
}



def _make_denorm(cfg):
    img_mean = cfg.data.get("img_mean", None)
    img_std = cfg.data.get("img_std", None)
    if img_mean is None or img_std is None:
        return None
    mean = np.array(img_mean, dtype=np.float32)
    std = np.array(img_std, dtype=np.float32)
    return lambda img: img * std + mean


def stage_input_space_viz(ctx: StageContext) -> None:
    metadata = DatasetMetadata.from_cfg(ctx.cfg)
    try:
        style.apply_paper_style(ctx.cfg)
    except Exception:
        pass

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
            projectors=["pca"],
            label_map={i: str(i) for i in range(metadata.ood_label)},
            ood_label=None,
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
            projectors=["pca"],
            label_map=metadata.label_map,
            ood_label=metadata.ood_label,
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
    textwidth = 6.0
    try:
        textwidth = float(ctx.cfg.viz.get("textwidth_in", textwidth))
    except Exception:
        pass

    max_cols = 2
    n_cols = min(len(loss_keys), max_cols)
    n_rows = (len(loss_keys) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(textwidth, style.FIG_H1 * n_rows),
        squeeze=False,
    )

    axes_flat = list(np.atleast_1d(axes).reshape(-1))
    for ax, key, color in zip(axes_flat, loss_keys, colors, strict=False):
        vals = [h[key] for h in history]
        ax.plot(epochs, vals, color=color, lw=2)
        ax.fill_between(epochs, vals, color=color, alpha=0.05)
        ax.set_title(key.capitalize(), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[len(loss_keys) :]:
        ax.axis("off")

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
    # compact version (for body)
    orig_n = ctx.cfg.viz.get("recon_n_samples", 2)
    ctx.cfg.viz["recon_n_samples"] = int(ctx.cfg.viz.get("recon_n_samples_compact", 2))
    _show_save_log(
        _build_reconstruction_fig_vae(ctx, metadata, compact=True),
        ctx,
        "reconstructions_compact.png",
        "eval/reconstructions_compact",
    )
    # full version (for appendix)
    ctx.cfg.viz["recon_n_samples"] = int(ctx.cfg.viz.get("recon_n_samples_full", 8))
    _show_save_log(
        _build_reconstruction_fig_vae(ctx, metadata, compact=False),
        ctx,
        "reconstructions_full.png",
        "eval/reconstructions_full",
    )
    ctx.cfg.viz["recon_n_samples"] = orig_n


def stage_reconstruction_viz_ddpm(ctx: StageContext) -> None:
    metadata = DatasetMetadata.from_cfg(ctx.cfg)
    for grid_mode, grid_suffix in [("compact", "_compact"), ("full", "_full")]:
        ctx.cfg.viz["ddpm_timestep_grid_mode"] = grid_mode
        _show_save_log(
            _build_ddpm_timestep_grid(ctx, metadata),
            ctx,
            f"ddpm_timestep_grid{grid_suffix}.png",
            f"eval/ddpm_timestep_grid{grid_suffix}",
        )
        ctx.cfg.viz["ddpm_denoising_trajectory_mode"] = grid_mode
        _show_save_log(
            _build_ddpm_denoising_trajectory(ctx, metadata),
            ctx,
            f"ddpm_denoising_trajectory{grid_suffix}.png",
            f"eval/ddpm_denoising_trajectory{grid_suffix}",
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
        fig = eval.plot_embeddings(
            np.concatenate([embeddings_id, embeddings_ood]),
            np.concatenate([labels_id, np.full(len(embeddings_ood), metadata.ood_label)]),
            ctx.cfg,
            title=f"{embed_label} — {metadata.id_name} (ID) vs {metadata.ood_name} (OOD)",
            label_map=metadata.label_map,
            ood_label=metadata.ood_label,
            run=ctx.run,
            wandb_key="eval/embedding_vs_ood",
            save_name="embedding_vs_ood",
        )
        plt.close(fig)

    if "vae" in str(ctx.cfg.model.model_type):
        raw_vecs, raw_labs, recon_vecs, recon_labs = _collect_recon_vectors(ctx)
        _plot_embedding_panel(
            ctx, raw_vecs, raw_labs, embeddings_id, labels_id, recon_vecs, recon_labs, embed_label
        )

    from src.evaluation.evaluate import _active_modes

    active = _active_modes(scores)
    aurocs = {mode: results.metrics[mode]["auroc"] for mode in results.metrics}
    plots_dir = Path(ctx.cfg.viz.plots_dir)
    plot_ood_evaluation(
        scores,
        aurocs,
        active,
        run=ctx.run,
        plots_dir=plots_dir,
        cfg=ctx.cfg,
        thresholds=results.thresholds,
        labels_map=MODE_LABELS,
    )

    ctx.run.log({"eval/status": "complete"})
    return results


def print_summary(cfg, results) -> None:
    model_type = cfg.model.model_type.upper()
    print("\n" + "=" * 75)
    print(f"          {model_type} OOD DETECTION — FINAL SUMMARY")
    print("=" * 75)
    print(f"  {'Mode':<26}   {'AUROC':>8}   {'AUPR':>8}   {'FPR@95':>8}")
    print("  " + "-" * 65)
    for mode, m in results.metrics.items():
        pretty_mode = MODE_LABELS.get(mode, mode)
        print(
            f"  {pretty_mode:<26}   {m['auroc']:>8.4f}   {m['aupr']:>8.4f}   {m['fpr_at_95_tpr']:>8.4f}"
        )
    print("-" * 75)
    print(json.dumps(results.to_dict(), indent=4))
    print("=" * 75)


def _build_toy_recon_fig(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
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

    fig, axes = plt.subplots(1, 2, figsize=(ctx.cfg.viz.textwidth_in, style.FIG_H1))
    kw = dict(s=14, alpha=0.7)
    axes[0].scatter(id_np[:, 0], id_np[:, 1], c="steelblue", label=f"{metadata.id_name} (ID)", **kw)
    axes[0].scatter(
        ood_np[:, 0], ood_np[:, 1], c="tomato", label=f"{metadata.ood_name} (OOD)", **kw
    )
    axes[0].set_title("Input space")
    axes[0].legend(markerscale=2, fontsize=9)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        recon_id_np[:, 0],
        recon_id_np[:, 1],
        c="steelblue",
        label=f"Recon {metadata.id_name} (ID)",
        **kw,
    )
    axes[1].scatter(
        recon_ood_np[:, 0],
        recon_ood_np[:, 1],
        c="darkorange",
        label=f"Recon {metadata.ood_name} (OOD)",
        **kw,
    )
    axes[1].set_title("Reconstructed space")
    axes[1].legend(markerscale=2, fontsize=9)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    ctx.model.train()
    return fig


def _build_reconstruction_fig_vae(
    ctx: StageContext, metadata: DatasetMetadata, compact: bool = True
) -> plt.Figure:
    ctx.model.eval()
    n_samples = int(ctx.cfg.viz.get("recon_n_samples", 2))
    x_id = next(iter(ctx.loaders["id_eval"]))[0][:n_samples].to(ctx.device)
    x_ood = next(iter(ctx.loaders["ood_eval"]))[0][:n_samples].to(ctx.device)
    with torch.no_grad():
        x_recon_id, _, _ = ctx.model(x_id)
        x_recon_ood, _, _ = ctx.model(x_ood)

    is_image = bool(ctx.cfg.data.get("is_image", False))
    textwidth = float(ctx.cfg.viz.get("textwidth_in", 6.0))
    denorm_fn = _make_denorm(ctx.cfg)

    if compact:
        fig, axes = plt.subplots(2, n_samples * 2, figsize=(textwidth, style.FIG_H1), squeeze=False)

        for col in range(n_samples):
            axes[0, col].set_ylabel("ID original", fontsize=9, fontweight="bold", color="steelblue")
            render_cell(axes[0, col], x_id[col].cpu(), is_image, "steelblue", denorm_fn)
            axes[1, col].set_ylabel(
                "ID reconstructed", fontsize=9, fontweight="bold", color="steelblue"
            )
            render_cell(axes[1, col], x_recon_id[col].cpu(), is_image, "steelblue", denorm_fn)

        for col in range(n_samples, n_samples * 2):
            idx = col - n_samples
            axes[0, col].set_ylabel("OOD original", fontsize=9, fontweight="bold", color="tomato")
            render_cell(axes[0, col], x_ood[idx].cpu(), is_image, "tomato", denorm_fn)
            axes[1, col].set_ylabel(
                "OOD reconstructed", fontsize=9, fontweight="bold", color="tomato"
            )
            render_cell(axes[1, col], x_recon_ood[idx].cpu(), is_image, "tomato", denorm_fn)

    else:
        fig, axes = plt.subplots(4, n_samples, figsize=(2.0 * n_samples, 2.0 * 4))

        for col in range(n_samples):
            # Bloque Superior: ID
            if col == 0:
                axes[0, col].set_ylabel(
                    "ID original", fontsize=10, fontweight="bold", color="steelblue"
                )
                axes[1, col].set_ylabel(
                    "ID reconstructed", fontsize=10, fontweight="bold", color="steelblue"
                )
            render_cell(axes[0, col], x_id[col].cpu(), is_image, "steelblue", denorm_fn)
            render_cell(axes[1, col], x_recon_id[col].cpu(), is_image, "steelblue", denorm_fn)

            # Bloque Inferior: OOD
            if col == 0:
                axes[2, col].set_ylabel(
                    "OOD original", fontsize=10, fontweight="bold", color="tomato"
                )
                axes[3, col].set_ylabel(
                    "OOD reconstructed", fontsize=10, fontweight="bold", color="tomato"
                )
            render_cell(axes[2, col], x_ood[col].cpu(), is_image, "tomato", denorm_fn)
            render_cell(axes[3, col], x_recon_ood[col].cpu(), is_image, "tomato", denorm_fn)

    plt.tight_layout()
    return fig


def _build_ddpm_timestep_grid(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
    ctx.model.eval()

    max_t = int(ctx.model.num_train_timesteps) - 1
    mode = str(ctx.cfg.viz.get("ddpm_timestep_grid_mode", "compact")).lower()
    is_image = bool(ctx.cfg.data.get("is_image", False))
    textwidth = float(ctx.cfg.viz.get("textwidth_in", 6.0))
    denorm_fn = _make_denorm(ctx.cfg)

    if mode == "full":
        keyframes = _ddpm_keyframes(max_t)
        x_id  = next(iter(ctx.loaders["id_eval"]))[0][: len(keyframes)].to(ctx.device)
        x_ood = next(iter(ctx.loaders["ood_eval"]))[0][: len(keyframes)].to(ctx.device)
        t_values = torch.tensor(keyframes, device=ctx.device, dtype=torch.long)[: x_id.shape[0]]
        noise_id  = ctx.model._fixed_noise(x_id.shape, ctx.device)
        noise_ood = ctx.model._fixed_noise(x_ood.shape, ctx.device)

        with torch.no_grad():
            _id_pairs  = [ctx.model.reconstruct_at_t(x_id[i:i+1],  t_values[i:i+1], noise=noise_id[i:i+1])  for i in range(len(keyframes))]
            _ood_pairs = [ctx.model.reconstruct_at_t(x_ood[i:i+1], t_values[i:i+1], noise=noise_ood[i:i+1]) for i in range(len(keyframes))]
            x_recon_id  = torch.stack([p[0][0] for p in _id_pairs])
            x_noisy_id  = torch.stack([p[1][0] for p in _id_pairs])
            x_recon_ood = torch.stack([p[0][0] for p in _ood_pairs])
            x_noisy_ood = torch.stack([p[1][0] for p in _ood_pairs])

        n_cols = len(keyframes) + 1
        fig, axes = plt.subplots(6, n_cols, figsize=(2.0 * n_cols, 1.8 * 6))
        rows_spec = [
            ("ID\nOriginal",    x_id,        "steelblue", None),
            ("ID\nNoisy (x_t)", x_noisy_id,  "steelblue", t_values),
            ("ID\nIterative",   x_recon_id,  "steelblue", t_values),
            ("OOD\nOriginal",   x_ood,       "tomato",    None),
            ("OOD\nNoisy (x_t)",x_noisy_ood, "tomato",    t_values),
            ("OOD\nIterative",  x_recon_ood, "tomato",    t_values),
        ]
        for row, (title, batch, color, t_vals) in enumerate(rows_spec):
            axes[row, 0].set_ylabel(title, fontsize=9, fontweight="bold", color=color)
            render_cell(axes[row, 0], batch[0].cpu(), is_image, color, denorm_fn)
            for col in range(len(keyframes)):
                ax = axes[row, col + 1]
                render_cell(ax, batch[col].cpu(), is_image, color, denorm_fn)
                if t_vals is not None:
                    ax.set_title(f"t={t_vals[col].item()}", fontsize=7, color=color, pad=2)

        fig.text(0.99, 0.75, "ID",  fontsize=16, fontweight="bold", color="steelblue", va="center", ha="right")
        fig.text(0.99, 0.25, "OOD", fontsize=16, fontweight="bold", color="tomato",    va="center", ha="right")
        plt.tight_layout()
        return fig

    # ── compact ──────────────────────────────────────────────────────────────
    compact_steps = [
        int(max_t * 0.05),
        int(max_t * 0.15),
        int(max_t * 0.25),
        int(max_t * 0.50),
        max_t,
    ]

    x_id  = next(iter(ctx.loaders["id_eval"]))[0][:1].to(ctx.device)
    x_ood = next(iter(ctx.loaders["ood_eval"]))[0][:1].to(ctx.device)
    noise_id  = ctx.model._fixed_noise(x_id.shape, ctx.device)
    noise_ood = ctx.model._fixed_noise(x_ood.shape, ctx.device)

    x_recon_id,  x_recon_ood = [], []

    with torch.no_grad():
        for t in compact_steps:
            t_tensor = torch.tensor([t], device=ctx.device, dtype=torch.long)
            rec_id,  _, _ = ctx.model.reconstruct_at_t(x_id,  t_tensor, noise=noise_id)
            rec_ood, _, _ = ctx.model.reconstruct_at_t(x_ood, t_tensor, noise=noise_ood)
            x_recon_id.append(rec_id[0].cpu())
            x_recon_ood.append(rec_ood[0].cpu())

    n_cols = len(compact_steps) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(textwidth, style.FIG_H2), squeeze=False)

    for col_i, t in enumerate(compact_steps):
        axes[0, col_i + 1].set_title(f"t={t}", fontsize=8)

    for row, (x_orig, recon_list, color, tag) in enumerate([
        (x_id,  x_recon_id,  "steelblue", "ID"),
        (x_ood, x_recon_ood, "tomato",    "OOD"),
    ]):
        render_cell(axes[row, 0], x_orig[0].cpu(), is_image, color, denorm_fn)
        axes[row, 0].set_ylabel(f"{tag}\nIterative", fontsize=9, fontweight="bold", color=color)
        for idx in range(len(compact_steps)):
            render_cell(axes[row, idx + 1], recon_list[idx], is_image, color, denorm_fn)

    plt.tight_layout()
    return fig


def _build_ddpm_denoising_trajectory(ctx: StageContext, metadata: DatasetMetadata) -> plt.Figure:
    ctx.model.eval()

    mode = str(ctx.cfg.viz.get("ddpm_denoising_trajectory_mode", "compact")).lower()
    is_image = bool(ctx.cfg.data.get("is_image", False))
    textwidth = float(ctx.cfg.viz.get("textwidth_in", 6.0))
    denorm_fn = _make_denorm(ctx.cfg)

    T = ctx.model.num_train_timesteps
    t_gen = T - 1

    if mode == "full":
        keyframes = _ddpm_keyframes(t_gen)
        n_runs = 4
        x_shape = next(iter(ctx.loaders["id_eval"]))[0][:1].to(ctx.device)
        trajs = []
        for seed in range(n_runs):
            rng = torch.Generator(device=ctx.device).manual_seed(seed)
            noise = torch.randn(x_shape.shape, generator=rng, device=ctx.device, dtype=x_shape.dtype)
            trajs.append(
                ctx.model.denoise_trajectory(
                    x_shape, t_start=t_gen, capture_timesteps=keyframes,
                    noise=noise, from_noise=True,
                )
            )

        ordered = sorted(keyframes, reverse=True)
        n_cols_full = len(ordered)
        fig, axes = plt.subplots(n_runs, n_cols_full, figsize=(2.0 * n_cols_full, 1.8 * n_runs))
        for r, traj in enumerate(trajs):
            axes[r, 0].set_ylabel(f"Run {r + 1}", fontsize=9, fontweight="bold")
            for c, t in enumerate(ordered):
                if t == t_gen:
                    token = "z ~ N(0,I)"
                elif t == 0:
                    token = "x̂₀"
                else:
                    token = f"t={t}"
                render_cell(axes[r, c], traj[t][0].cpu(), is_image, "steelblue", denorm_fn)
                axes[r, c].set_title(token, fontsize=7)
        plt.tight_layout()
        return fig

    # ── compact ──────────────────────────────────────────────────────────────
    capture_timesteps = [t_gen, int(t_gen * 0.75), int(t_gen * 0.50), int(t_gen * 0.25), 0]
    x_shape = next(iter(ctx.loaders["id_eval"]))[0][:1].to(ctx.device)

    trajs = []
    for seed in range(2):
        rng = torch.Generator(device=ctx.device).manual_seed(seed)
        noise = torch.randn(x_shape.shape, generator=rng, device=ctx.device, dtype=x_shape.dtype)
        trajs.append(
            ctx.model.denoise_trajectory(
                x_shape, t_start=t_gen, capture_timesteps=capture_timesteps,
                noise=noise, from_noise=True,
            )
        )

    fig, axes = plt.subplots(2, len(capture_timesteps), figsize=(textwidth, style.FIG_H1), squeeze=False)

    for row, traj in enumerate(trajs):
        axes[row, 0].set_ylabel(f"Run {row + 1}", fontsize=9, fontweight="bold")
        for c, t in enumerate(capture_timesteps):
            if t == t_gen:
                token = "z ~ N(0,I)"
            elif t == 0:
                token = "x̂₀"
            else:
                token = f"x_t  (t={t})"
            render_cell(axes[row, c], traj[int(t)][0].cpu(), is_image, "steelblue", denorm_fn)
            axes[row, c].set_title(token, fontsize=8)

    plt.tight_layout()
    return fig


def _ddpm_keyframes(max_t: int, n: int = 7) -> list[int]:
    T = max_t + 1
    return [max(1, min(int(round(T * i / (n - 1))), max_t)) for i in range(n)]


def _show_save_log(
    fig,
    ctx: StageContext,
    filename: str,
    wandb_key: str,
    save_vector_formats: list[str] | None = None,
) -> None:
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
        save_vector_formats=save_vector_formats,
        png_dpi=int(ctx.cfg.viz.get("savefig_dpi", 300)),
    )
    plt.close(fig)


def _collect_raw_vectors(loader):
    vecs, labs = [], []
    for x, y in loader:
        vecs.append(x.view(x.size(0), -1).cpu().numpy())
        labs.append(y.cpu().numpy())
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
    cmap = plt.colormaps["tab10"].resampled(10)
    textwidth = float(ctx.cfg.viz.get("textwidth_in", 6.0))
    fig, axs = plt.subplots(3, 1, figsize=(textwidth, style.FIG_H3), squeeze=False)
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
            axs[panel, 0].scatter(
                coords[m, 0], coords[m, 1], c=[cmap(c)], s=8, alpha=0.5, label=str(c)
            )
        axs[panel, 0].set_title(title, fontsize=10, fontweight="bold")
        axs[panel, 0].axis("off")

    axs[0, 0].legend(markerscale=2, fontsize=8, ncol=2)
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
        png_dpi=int(ctx.cfg.viz.get("savefig_dpi", 300)),
    )
    plt.close(fig)


def _build_scores_payload(scores) -> dict:
    payload = {}
    for mode in MODE_LABELS.keys():
        id_arr = getattr(scores, f"id_{mode}", np.array([]))
        ood_arr = getattr(scores, f"ood_{mode}", np.array([]))
        if id_arr.size > 0:
            payload[f"scores/id_{mode}_mean"] = float(id_arr.mean())
            payload[f"scores/ood_{mode}_mean"] = float(ood_arr.mean())
    return payload

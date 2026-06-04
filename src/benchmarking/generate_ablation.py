from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import save_figure

_DDPM_SCORE_MODES = [
    "noise_single", "noise_multi_mse", "noise_multi_cosine",
    "recon_single", "recon_multi",
]
_MODE_COLORS = {
    "noise_single": "steelblue",
    "noise_multi_mse": "darkorange",
    "noise_multi_cosine": "mediumseagreen",
    "recon_single": "mediumpurple",
    "recon_multi": "crimson",
}
_MODE_LABELS = {
    "noise_single": "Single-step MSE",
    "noise_multi_mse": "Multi-step MSE (z-score)",
    "noise_multi_cosine": "Multi-step Cosine (z-score)",
    "recon_single": "Single-step Reconstruction",
    "recon_multi": "Multi-step Reconstruction",
}

# Groups of datasets that share a figure and y-axis limits
_DATASET_GROUPS = {
    "sicap": ["sicap_c1", "sicap_c12"],
    "pathmnist": ["pathmnist_c1", "pathmnist_c2"],
}


def _parse_t(experiment_id: str) -> int | None:
    m = re.search(r"_t(\d+)(?:_|$)", str(experiment_id))
    return int(m.group(1)) if m else None


def _load_ablation_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("method", "dataset", "score"):
        df[col] = df[col].astype(str).str.strip().str.lower()

        def _normalize_score(val: str) -> str:
            if "recon_single" in val or "single-step reconstruction" in val:
                return "recon_single"
            if "recon_multi" in val or "multi-step reconstruction" in val:
                return "recon_multi"
            if "single" in val or "noise_single" in val:
                return "noise_single"
            if "cosine" in val or "noise_multi_cosine" in val:
                return "noise_multi_cosine"
            if "mse" in val or "noise_multi_mse" in val:
                return "noise_multi_mse"
            return val

        df["score"] = df["score"].apply(_normalize_score)

    for col in ("auroc", "aupr", "fpr_at_95_tpr", "seed", "lr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["t_steps"] = df["experiment_id"].apply(_parse_t)

    all_key_datasets = [d for group in _DATASET_GROUPS.values() for d in group]
    mask = (
        df["method"].str.contains("ddpm")
        & df["dataset"].isin(all_key_datasets)
        & df["t_steps"].notna()
        & df["score"].isin(_DDPM_SCORE_MODES)
        & df["auroc"].notna()
    )
    return df[mask].copy()


def _write_ablation_tex(
    grouped_all: pd.DataFrame, t_vals_all: list[int], datasets: list[str], out: Path, suffix: str
) -> None:
    n_cols = 1 + len(t_vals_all)
    col_spec = "l" + "c" * len(t_vals_all)
    lines = [
        rf"\begin{{longtable}}{{{col_spec}}}",
        r"\toprule",
        "Score Mode & " + " & ".join(f"$T={t}$" for t in t_vals_all) + r" \\",
        r"\midrule",
    ]
    for i, dataset in enumerate(datasets):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{dataset.upper()}}}}} \\")
        lines.append(r"\midrule")
        sub = grouped_all[grouped_all["dataset"] == dataset]
        for mode in _DDPM_SCORE_MODES:
            mode_data = sub[sub["score"] == mode].set_index("T")["AUROC"]
            cells = [f"{mode_data[t]:.4f}" if t in mode_data.index else "--" for t in t_vals_all]
            lines.append(_MODE_LABELS[mode] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    tex_path = out / f"table_ablation_t_{suffix}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_path}")


def _plot_ablation_group(
    abl: pd.DataFrame, datasets: list[str], out: Path, suffix: str, group_title: str
) -> None:
    """Bar plot AUROC vs T for one group of datasets with shared y-axis limits."""
    # Pre-compute global y_min / y_max across all datasets in this group
    all_stats = abl[abl["dataset"].isin(datasets)].groupby(["score", "t_steps"])["auroc"].agg(
        ["mean", "std"]
    )
    valid_means = all_stats["mean"].dropna()
    if valid_means.empty:
        return
    g_min = float(valid_means.min())
    g_max = float(valid_means.max())
    pad = 0.05 * (g_max - g_min) if g_max > g_min else 0.05
    shared_ymin = max(0.0, g_min - pad)
    shared_ymax = min(1.0, g_max + pad)

    textwidth = 5.65
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(textwidth * len(datasets), textwidth * 0.55), squeeze=False
    )
    fig.suptitle(
        f"DDPM — AUROC vs. scoring steps (T) [{group_title}]",
        fontsize=9,
        fontweight="normal",
        y=0.97,
    )

    for ax, dataset in zip(axes[0], datasets, strict=True):
        subset = abl[abl["dataset"] == dataset]
        t_vals = sorted(subset["t_steps"].dropna().unique().astype(int))
        stats = subset.groupby(["score", "t_steps"])["auroc"].agg(["mean", "std"]).reset_index()

        x = np.arange(len(t_vals))
        bar_width = 0.22
        offsets = np.linspace(-bar_width, bar_width, len(_DDPM_SCORE_MODES))

        for offset, mode in zip(offsets, _DDPM_SCORE_MODES, strict=True):
            mode_stats = stats[stats["score"] == mode].set_index("t_steps")
            heights = [
                float(mode_stats.loc[t, "mean"]) if t in mode_stats.index else np.nan
                for t in t_vals
            ]
            errors = [
                float(mode_stats.loc[t, "std"]) if t in mode_stats.index else 0.0 for t in t_vals
            ]
            ax.bar(
                x + offset,
                heights,
                width=bar_width,
                yerr=errors,
                label=_MODE_LABELS[mode],
                color=_MODE_COLORS[mode],
                capsize=3,
                alpha=0.95,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(dataset.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("n_score_steps (T)")
        ax.set_ylabel("AUROC")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in t_vals])
        ax.set_ylim(shared_ymin, shared_ymax)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = out / f"ablation_t_auroc_{suffix}.png"
    save_figure(
        fig,
        fig_path,
        run=None,
        image_key=None,
        artifact_prefix=f"ablation-{suffix}",
        png_dpi=300,
    )
    plt.close(fig)
    print(f"Saved → {fig_path} (and PDF)")


def run_t_ablation(
    csv_path: str = "results/summary/comparison.csv",
    out_dir: str = "results/summary",
) -> None:
    """Fig 12: AUROC vs T line plot. Tab 13: T ablation table (markdown + LaTeX).

    Produces one figure + table per dataset group (sicap, pathmnist).
    Within each figure all subplots share the same y-axis lower bound.
    """
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    abl = _load_ablation_df(src)
    if abl.empty:
        print("No T-ablation data found (need DDPM key-dataset experiments with varying T).")
        return

    # single-step modes have a fixed T — assign T=1 for plotting
    abl.loc[abl["score"].isin(["noise_single", "recon_single"]), "t_steps"] = 1

    for group_name, candidate_datasets in _DATASET_GROUPS.items():
        datasets = [d for d in candidate_datasets if d in abl["dataset"].values]
        if not datasets:
            continue

        group_abl = abl[abl["dataset"].isin(datasets)].copy()

        _plot_ablation_group(group_abl, datasets, out, suffix=group_name, group_title=group_name.upper())

        grouped_all = (
            group_abl.groupby(["dataset", "score", "t_steps"])["auroc"]
            .mean()
            .reset_index()
            .rename(columns={"t_steps": "T", "auroc": "AUROC"})
        )
        t_vals_all = sorted(group_abl["t_steps"].dropna().unique().astype(int))
        _write_ablation_tex(grouped_all, t_vals_all, datasets, out, suffix=group_name)

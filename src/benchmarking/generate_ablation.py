from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import save_figure

_DDPM_SCORE_MODES = [
    "noise_single",
    "noise_multi_mse",
    "noise_multi_cosine",
    "recon_single",
    "recon_multi",
]
_MODE_COLORS = {
    "noise_single": "steelblue",
    "noise_multi_mse": "royalblue",
    "noise_multi_cosine": "cornflowerblue",
    "recon_single": "tomato",
    "recon_multi": "crimson",
}
_MODE_LABELS = {
    "noise_single": "Noise Single-step",
    "noise_multi_mse": "Noise Multi-step MSE",
    "noise_multi_cosine": "Noise Multi-step Cosine",
    "recon_single": "Recon Single-step",
    "recon_multi": "Recon Multi-step",
}


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
            mode_sub = sub[sub["score"] == mode].set_index("T")
            mode_means = {
                t: mode_sub.loc[t, "AUROC"]
                for t in t_vals_all
                if t in mode_sub.index and pd.notna(mode_sub.loc[t, "AUROC"])
            }
            best_t = max(mode_means, key=mode_means.get) if mode_means else None
            cells = []
            for t in t_vals_all:
                if t not in mode_sub.index:
                    cells.append("--")
                    continue
                mean = mode_sub.loc[t, "AUROC"]
                std = mode_sub.loc[t, "AUROC_std"]
                if pd.notna(std) and std > 0:
                    if t == best_t:
                        cell = f"$\\mathbf{{{mean:.4f}}} \\pm \\mathbf{{{std:.4f}}}$"
                    else:
                        cell = f"${mean:.4f} \\pm {std:.4f}$"
                else:
                    if t == best_t:
                        cell = f"\\textbf{{{mean:.4f}}}"
                    else:
                        cell = f"{mean:.4f}"
                cells.append(cell)
            lines.append(_MODE_LABELS[mode] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    tex_path = out / f"table_ablation_t_{suffix}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_path}")


_CHART_MODES = ["noise_single", "recon_single", "noise_multi_mse", "recon_multi"]
_CHART_YMIN = 0.75


def _plot_ablation_group(
    abl: pd.DataFrame, datasets: list[str], out: Path, suffix: str, group_title: str
) -> None:
    chart_abl = abl[abl["score"].isin(_CHART_MODES)]
    valid_means = (
        chart_abl[chart_abl["dataset"].isin(datasets)]
        .groupby(["score", "t_steps"])["auroc"]
        .mean()
        .dropna()
    )
    if valid_means.empty:
        return
    g_max = float(valid_means.max())
    shared_ymax = min(1.0, g_max + 0.02)

    textwidth = 5.65
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(textwidth * len(datasets), textwidth * 0.55), squeeze=False
    )

    bar_width = 0.4  # bars in a pair touch: offset between centers = bar_width

    for ax, dataset in zip(axes[0], datasets, strict=True):
        subset = abl[abl["dataset"] == dataset]
        t_vals = sorted(subset["t_steps"].dropna().unique().astype(int))
        stats = (
            subset[subset["score"].isin(_CHART_MODES)]
            .groupby(["score", "t_steps"])["auroc"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = np.arange(len(t_vals))
        legend_added: set[str] = set()

        for t_idx, t in enumerate(t_vals):
            t_stats = stats[stats["t_steps"] == t]
            active = [
                m
                for m in _CHART_MODES
                if (t_stats["score"] == m).any()
                and pd.notna(t_stats.loc[t_stats["score"] == m, "mean"].values[0])
            ]
            n = len(active)
            if n == 0:
                continue
            offsets = (np.arange(n) - (n - 1) / 2) * bar_width

            for i, mode in enumerate(active):
                row = t_stats[t_stats["score"] == mode]
                mean = float(row["mean"].values[0])
                std_val = row["std"].values[0]
                err = float(std_val) if pd.notna(std_val) and std_val > 0 else None
                label = _MODE_LABELS[mode] if mode not in legend_added else None
                legend_added.add(mode)
                ax.bar(
                    x[t_idx] + offsets[i],
                    mean,
                    width=bar_width,
                    yerr=err,
                    label=label,
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
        ax.set_ylim(_CHART_YMIN, shared_ymax)
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
    # Fig 12: AUROC vs T line plot. Tab 13: T ablation table
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    abl = _load_ablation_df(src)
    if abl.empty:
        print("No T-ablation data found (need DDPM key-dataset experiments with varying T).")
        return

    # single-step modes as T=1 for plotting
    abl.loc[abl["score"].isin(["noise_single", "recon_single"]), "t_steps"] = 1

    for group_name, candidate_datasets in _DATASET_GROUPS.items():
        datasets = [d for d in candidate_datasets if d in abl["dataset"].values]
        if not datasets:
            continue

        group_abl = abl[abl["dataset"].isin(datasets)].copy()

        _plot_ablation_group(
            group_abl, datasets, out, suffix=group_name, group_title=group_name.upper()
        )

        grouped_all = (
            group_abl.groupby(["dataset", "score", "t_steps"])["auroc"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"t_steps": "T", "mean": "AUROC", "std": "AUROC_std"})
        )
        t_vals_all = sorted(group_abl["t_steps"].dropna().unique().astype(int))
        _write_ablation_tex(grouped_all, t_vals_all, datasets, out, suffix=group_name)

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import save_figure

_DDPM_SCORE_MODES = ["noise_single", "noise_multi_mse", "noise_multi_cosine"]
_MODE_COLORS = {
    "noise_single": "steelblue",
    "noise_multi_mse": "darkorange",
    "noise_multi_cosine": "mediumseagreen",
}
_MODE_LABELS = {
    "noise_single": "Single-step MSE",
    "noise_multi_mse": "Multi-step MSE (z-score)",
    "noise_multi_cosine": "Multi-step Cosine (z-score)",
}


def _parse_t(experiment_id: str) -> int | None:
    m = re.search(r"_t(\d+)(?:_|$)", str(experiment_id))
    return int(m.group(1)) if m else None


def _load_ablation_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("method", "dataset", "score"):
        df[col] = df[col].astype(str).str.strip().str.lower()

        def _normalize_score(val: str) -> str:
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

    mask = (
        df["method"].str.contains("ddpm")
        & df["dataset"].isin(["sicap_c1", "sicap_c12"])
        & df["t_steps"].notna()
        & df["score"].isin(_DDPM_SCORE_MODES)
        & df["auroc"].notna()
    )
    return df[mask].copy()


def _write_ablation_tex(
    grouped_all: pd.DataFrame, t_vals_all: list[int], datasets: list[str], out: Path
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
    tex_path = out / "table_ablation_t.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_path}")


def run_t_ablation(
    csv_path: str = "results/summary/comparison.csv",
    out_dir: str = "results/summary",
) -> None:
    """Fig 12: AUROC vs T line plot. Tab 13: T ablation table (markdown + LaTeX)."""
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    abl = _load_ablation_df(src)
    if abl.empty:
        print("No T-ablation data found (need DDPM SICAP experiments with varying T).")
        return

    # ── CAMBIO CRÍTICO Y GLOBAL ──────────────────────────────────────────────
    abl.loc[abl["score"] == "noise_single", "t_steps"] = 1

    datasets = sorted(abl["dataset"].unique())

    # ── Fig 12: AUROC vs T ────────────────────────────────────────────────────
    textwidth = 5.65
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(textwidth * len(datasets), textwidth * 0.55), squeeze=False
    )
    fig.suptitle("DDPM — AUROC vs. scoring steps (T)", fontsize=9, fontweight="normal", y=0.97)

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
        ax.set_xticklabels([str(t) for t in t_vals])  # El primer tick será "1"

        valid_means = stats["mean"].dropna()
        y_min = float(valid_means.min()) if not valid_means.empty else 0.0
        y_max = float(valid_means.max()) if not valid_means.empty else 1.0
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = out / "ablation_t_auroc.png"
    save_figure(
        fig,
        fig_path,
        run=None,
        image_key=None,
        artifact_prefix="ablation",
        png_dpi=300,
    )
    plt.close(fig)
    print(f"Saved → {fig_path} (and PDF)")

    # ── Tab 13: T ablation table ──────────────────────────────────────────────
    grouped_all = (
        abl.groupby(["dataset", "score", "t_steps"])["auroc"]
        .mean()
        .reset_index()
        .rename(columns={"t_steps": "T", "auroc": "AUROC"})
    )
    t_vals_all = sorted(abl["t_steps"].dropna().unique().astype(int))

    _write_ablation_tex(grouped_all, t_vals_all, datasets, out)

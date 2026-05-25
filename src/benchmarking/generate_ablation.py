from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_DDPM_SCORE_MODES = ["noise_single", "noise_multi_mse", "noise_multi_cosine"]
_MODE_COLORS = {
    "noise_single":      "steelblue",
    "noise_multi_mse":   "darkorange",
    "noise_multi_cosine": "mediumseagreen",
}
_MODE_LABELS = {
    "noise_single":      "Single-step MSE",
    "noise_multi_mse":   "Multi-step MSE (z-score)",
    "noise_multi_cosine": "Multi-step Cosine (z-score)",
}


def _parse_t(experiment_id: str) -> int | None:
    m = re.search(r"_t(\d+)(?:_|$)", str(experiment_id))
    return int(m.group(1)) if m else None


def _load_ablation_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("method", "dataset", "score"):
        df[col] = df[col].astype(str).str.strip().str.lower()
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
        rf"\begin{{tabular}}{{{col_spec}}}",
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
            cells = [
                f"{mode_data[t]:.4f}" if t in mode_data.index else "--"
                for t in t_vals_all
            ]
            lines.append(_MODE_LABELS[mode] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
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

    datasets = sorted(abl["dataset"].unique())

    # ── Fig 12: AUROC vs T ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    fig.suptitle(
        "DDPM — AUROC vs n_score_steps (T) en SICAP",
        fontsize=14, fontweight="bold",
    )

    for ax, dataset in zip(axes[0], datasets):
        grouped = (
            abl[abl["dataset"] == dataset]
            .groupby(["score", "t_steps"])["auroc"]
            .mean()
            .reset_index()
        )
        t_vals = sorted(abl["t_steps"].dropna().unique().astype(int))

        for mode in _DDPM_SCORE_MODES:
            mode_data = grouped[grouped["score"] == mode].sort_values("t_steps")
            if mode_data.empty:
                continue
            ax.plot(
                mode_data["t_steps"],
                mode_data["auroc"],
                marker="o",
                label=_MODE_LABELS[mode],
                color=_MODE_COLORS[mode],
                lw=2,
                markersize=7,
            )

        ax.set_title(dataset.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("n_score_steps (T)")
        ax.set_ylabel("AUROC")
        ax.set_xticks(t_vals)
        aurocs = grouped["auroc"].dropna()
        if not aurocs.empty:
            y_min, y_max = float(aurocs.min()), float(aurocs.max())
            pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out / "ablation_t_auroc.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {fig_path}")

    # ── Tab 13: T ablation table ──────────────────────────────────────────────
    grouped_all = (
        abl.groupby(["dataset", "score", "t_steps"])["auroc"]
        .mean()
        .reset_index()
        .rename(columns={"t_steps": "T", "auroc": "AUROC"})
    )
    t_vals_all = sorted(abl["t_steps"].dropna().unique().astype(int))

    _write_ablation_tex(grouped_all, t_vals_all, datasets, out)

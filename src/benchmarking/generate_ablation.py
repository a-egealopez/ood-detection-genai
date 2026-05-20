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


def run_t_ablation(
    csv_path: str = "results/summary/comparison.csv",
    out_dir: str = "results/summary",
) -> None:
    """Fig 12: AUROC vs T line plot. Tab 13: T ablation markdown table."""
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
        ax.set_ylim(0.4, 1.0)
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

    lines = ["# DDPM — Ablación de T (n_score_steps)\n"]
    for dataset in datasets:
        lines.append(f"## {dataset.upper()}\n")
        sub = grouped_all[grouped_all["dataset"] == dataset]
        if sub.empty:
            lines.append("Sin datos.\n")
            continue

        t_cols = [f"T={t}" for t in t_vals_all]
        header = "| Score Mode | " + " | ".join(t_cols) + " |"
        sep    = "|------------|" + "|".join("------" for _ in t_cols) + "|"
        lines += [header, sep]

        for mode in _DDPM_SCORE_MODES:
            mode_data = sub[sub["score"] == mode].set_index("T")["AUROC"]
            cells = [
                f"{mode_data[t]:.4f}" if t in mode_data.index else "-"
                for t in t_vals_all
            ]
            lines.append(f"| {_MODE_LABELS[mode]} | " + " | ".join(cells) + " |")

        lines.append("")

    tab_path = out / "table_ablation_t.md"
    tab_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tab_path}")

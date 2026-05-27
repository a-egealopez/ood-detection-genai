from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.artifacts import save_figure

REQUIRED_COLUMNS = [
    "experiment_id",
    "method",
    "dataset",
    "seed",
    "lr",
    "score",
    "auroc",
    "aupr",
    "fpr_at_95_tpr",
    "threshold_at_5_fpr",
    "tpr_at_5_fpr",
]

METHOD_ALIASES = {"vae_toy": "vae", "ddpm_toy": "ddpm", "vae_path": "vae", "ddpm_path": "ddpm"}

GENERATIVE_METHODS = {"vae", "ddpm"}
DISTANCE_METHODS = {"knn", "mahalanobis"}


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"comparison.csv is missing columns: {missing}")
    df = df.copy()
    for col in ("lr", "seed", "auroc", "aupr", "fpr_at_95_tpr", "threshold_at_5_fpr", "tpr_at_5_fpr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("method", "dataset", "score"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["method"] = df["method"].replace(METHOD_ALIASES)
    df = df.dropna(subset=["auroc"])
    df = df[(df["auroc"] >= 0.0) & (df["auroc"] <= 1.0)]
    return df


def _fmt(value, fmt: str, fallback: str = "-") -> str:
    return fallback if pd.isna(value) else format(value, fmt)


def _escape_tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _tex_meanstd(mean, std, fmt: str = ".4f", fallback: str = "--") -> str:
    if pd.isna(mean):
        return fallback
    s = format(mean, fmt)
    return f"{s} $\\pm$ {format(std, fmt)}" if not pd.isna(std) and std > 0 else s


def _parse_t_from_id(experiment_id: str) -> int | None:
    import re
    m = re.search(r"_t(\d+)(?:_|$)", str(experiment_id))
    return int(m.group(1)) if m else None


def _write_tab10_tex(agg: pd.DataFrame, sicap_datasets: list[str], out: Path) -> None:
    n_cols = 8
    lines = [
        r"\begin{tabular}{llrllccc}",
        r"\toprule",
        r"Método & LR & T & Score & N & AUROC & AUPR & FPR@95\% \\",
        r"\midrule",
    ]
    for i, dataset in enumerate(sicap_datasets):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{dataset.upper()}}}}} \\")
        lines.append(r"\midrule")
        sub = agg[agg["dataset"] == dataset].sort_values(["method", "lr", "score", "t_steps"])
        for _, r in sub.iterrows():
            t_str = str(int(r["t_steps"])) if pd.notna(r["t_steps"]) else "--"
            n = int(r["auroc_count"]) if pd.notna(r["auroc_count"]) else 1
            cells = [
                _escape_tex(r["method"]),
                _fmt(r["lr"], "g"),
                t_str,
                _escape_tex(r["score"]),
                str(n),
                _tex_meanstd(r["auroc_mean"], r["auroc_std"]),
                _tex_meanstd(r["aupr_mean"], r["aupr_std"]),
                _tex_meanstd(r["fpr_at_95_tpr_mean"], r["fpr_at_95_tpr_std"]),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = out / "table_results_sicap.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_path}")


def _bold_if_best(value: str, metric: float, best: float) -> str:
    return rf"\textbf{{{value}}}" if pd.notna(metric) and metric == best else value


def _write_tab11_tex(
    agg: pd.DataFrame, dist: pd.DataFrame, sicap_datasets: list[str], out: Path
) -> None:
    n_cols = 9
    lines = [
        r"\begin{tabular}{lllrllccc}",
        r"\toprule",
        r"Método & LR & T & Score & Seed & N & AUROC & AUPR & FPR@95\% \\",
        r"\midrule",
    ]
    for i, dataset in enumerate(sicap_datasets):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{dataset.upper()}}}}} \\")
        lines.append(r"\midrule")

        rows_to_write = []
        for meth in ("knn", "mahalanobis"):
            rows = dist[(dist["dataset"] == dataset) & (dist["method"] == meth)]
            if rows.empty:
                continue
            best = rows.loc[rows["auroc"].idxmax()]
            rows_to_write.append(
                {
                    "method": meth,
                    "lr": "--",
                    "t": "--",
                    "score": "--",
                    "seed": _fmt(best["seed"], ".0f"),
                    "n": "1",
                    "auroc": float(best["auroc"]),
                    "aupr": float(best["aupr"]),
                    "fpr": float(best["fpr_at_95_tpr"]),
                }
            )
        for meth in ("vae", "ddpm"):
            sub_agg = agg[(agg["dataset"] == dataset) & (agg["method"] == meth)]
            if sub_agg.empty:
                continue
            b = sub_agg.loc[sub_agg["auroc_mean"].idxmax()]
            rows_to_write.append(
                {
                    "method": meth,
                    "lr": _fmt(b["lr"], "g"),
                    "t": str(int(b["t_steps"])) if pd.notna(b["t_steps"]) else "--",
                    "score": _escape_tex(b["score"]),
                    "seed": _fmt(b["seed"], ".0f"),
                    "n": str(int(b["auroc_count"])),
                    "auroc": float(b["auroc_mean"]),
                    "aupr": float(b["aupr_mean"]),
                    "fpr": float(b["fpr_at_95_tpr_mean"]),
                }
            )

        best_auroc = max((row["auroc"] for row in rows_to_write), default=float("nan"))
        for row in rows_to_write:
            cells = [
                _escape_tex(row["method"]),
                _escape_tex(row["lr"]),
                _escape_tex(row["t"]),
                _escape_tex(row["score"]),
                _escape_tex(row["seed"]),
                row["n"],
                _bold_if_best(_fmt(row["auroc"], ".4f"), row["auroc"], best_auroc),
                _fmt(row["aupr"], ".4f"),
                _fmt(row["fpr"], ".4f"),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = out / "table_comparison_sicap.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_path}")


def _plot_sicap_methods(
    agg: pd.DataFrame, dist: pd.DataFrame, sicap_datasets: list[str], out: Path
) -> None:
    methods = ["knn", "mahalanobis", "vae", "ddpm"]
    colors = {
        "knn": "gray",
        "mahalanobis": "black",
        "vae": "steelblue",
        "ddpm": "darkorange",
    }

    fig, axes = plt.subplots(
        1,
        len(sicap_datasets),
        figsize=(6 * len(sicap_datasets), 4),
        squeeze=False,
    )
    for ax, dataset in zip(axes[0], sicap_datasets):
        values = []
        errors = []
        labels = []
        for meth in methods:
            if meth in DISTANCE_METHODS:
                rows = dist[(dist["dataset"] == dataset) & (dist["method"] == meth)]
                if rows.empty:
                    values.append(float("nan"))
                    errors.append(0.0)
                    labels.append(meth)
                    continue
                best = rows.loc[rows["auroc"].idxmax()]
                values.append(float(best["auroc"]))
                errors.append(0.0)
                labels.append(meth)
            else:
                sub_agg = agg[(agg["dataset"] == dataset) & (agg["method"] == meth)]
                if sub_agg.empty:
                    values.append(float("nan"))
                    errors.append(0.0)
                    labels.append(meth)
                    continue
                b = sub_agg.loc[sub_agg["auroc_mean"].idxmax()]
                values.append(float(b["auroc_mean"]))
                errors.append(float(b["auroc_std"]))
                labels.append(meth)

        x = range(len(methods))
        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            color=[colors[m] for m in methods],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("AUROC")
        ax.set_title(dataset.upper(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    save_figure(
        fig,
        out / "sicap_methods_comparison.png",
        run=None,
        image_key=None,
        artifact_prefix="sicap-methods-comparison",
    )
    plt.close(fig)


def run_aggregate_tables(
    csv_path: str = "results/summary/comparison.csv",
    out_dir: str = "results/summary",
) -> None:
    """Tab 10: mean±std per (method, lr, score) across seeds.
    Tab 11: best-per-method comparison across all configs.
    """
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    df = _validate(pd.read_csv(src))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df["t_steps"] = df["experiment_id"].apply(_parse_t_from_id)

    sicap_datasets = [d for d in ("sicap_c1", "sicap_c12") if d in df["dataset"].values]
    if not sicap_datasets:
        print("No SICAP data found in CSV. Run exp 3 first.")
        return

    gen_mask = df["method"].isin(GENERATIVE_METHODS) & df["dataset"].isin(sicap_datasets)
    gen = df[gen_mask].copy()

    group_keys = ["dataset", "method", "lr", "t_steps", "score"]
    agg = (
        gen.groupby(group_keys, dropna=False)[["auroc", "aupr", "fpr_at_95_tpr"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    _write_tab10_tex(agg, sicap_datasets, out)

    dist_mask = df["method"].isin(DISTANCE_METHODS) & df["dataset"].isin(sicap_datasets)
    dist = df[dist_mask].copy()

    _write_tab11_tex(agg, dist, sicap_datasets, out)
    _plot_sicap_methods(agg, dist, sicap_datasets, out)

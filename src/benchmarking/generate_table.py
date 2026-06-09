from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

_SCORE_DISPLAY = {
    "recon": "MSE Recon.",
    "elbo": "ELBO",
    "latent_knn": "Latent KNN",
    "latent_mah": "Latent Mah.",
    "noise_single": "Single-step",
    "noise_multi_mse": "Multi-step MSE",
    "noise_multi_cosine": "Multi-step Cosine",
    "recon_single": "Single-step Reconstruction",
    "recon_multi": "Multi-step Reconstruction",
    "residual_mah": "Residual Mah.",
    "residual_knn": "Residual KNN",
}

_METHOD_DISPLAY = {
    "vae": "VAE",
    "ddpm": "DDPM",
    "knn": "KNN",
    "mahalanobis": "Mahalanobis",
}

# Groups of datasets treated together (one figure + table per group)
_DATASET_GROUPS = {
    "sicap": ["sicap_c1", "sicap_c12"],
    "pathmnist": ["pathmnist_c1", "pathmnist_c2"],
}


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"comparison.csv is missing columns: {missing}")
    df = df.copy()
    for col in (
        "lr",
        "seed",
        "auroc",
        "aupr",
        "fpr_at_95_tpr",
        "threshold_at_5_fpr",
        "tpr_at_5_fpr",
    ):
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


def _write_tab_results_tex(
    agg: pd.DataFrame, datasets: list[str], out: Path, suffix: str,
    caption: str = "", label: str = "",
) -> None:
    groups = {"Baselines": ["knn", "mahalanobis"], "VAE": ["vae"], "DDPM": ["ddpm"]}

    n_cols = 6
    cap_line = rf"    \caption{{{caption}}} \label{{{label}}} \\" if caption else ""
    lines = [
        r"\begin{longtable}{llllll}",
        *(([cap_line]) if cap_line else []),
        r"    \toprule",
        r"    Método & LR & Score & AUROC & AUPR & FPR@95\% \\",
        r"    \midrule",
        r"    \endfirsthead",
        r"    ",
        r"    \toprule",
        r"    Método & LR & Score & AUROC & AUPR & FPR@95\% \\",
        r"    \midrule",
        r"    \endhead",
        r"    ",
        r"    \midrule",
        r"    \multicolumn{6}{r}{\textit{Continúa en la siguiente página}} \\",
        r"    \endfoot",
        r"    ",
        r"    \bottomrule",
        r"    \endlastfoot",
    ]

    for dataset in datasets:
        ds_display = dataset.upper().replace("_", r"\_")
        lines.append(rf"    \multicolumn{{{n_cols}}}{{l}}{{\textbf{{{ds_display}}}}} \\")
        lines.append(r"    \midrule")

        sub = agg[agg["dataset"] == dataset]

        for group_name, methods in groups.items():
            lines.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\cellcolor{{gray!10}}\textbf{{{group_name}}}}} \\"
            )

            group_data = sub[sub["method"].isin(methods)].sort_values(["method", "lr", "score"])
            best_auroc = group_data["auroc_mean"].max()

            for _, r in group_data.iterrows():
                if group_name == "DDPM" and (pd.isna(r["auroc_std"]) or pd.isna(r["aupr_std"])):
                    continue

                auroc_val = r["auroc_mean"]
                auroc_str = _tex_meanstd(auroc_val, r["auroc_std"])
                if pd.notna(auroc_val) and auroc_val == best_auroc:
                    auroc_str = rf"\textbf{{{auroc_str}}}"

                cells = [
                    _escape_tex(_METHOD_DISPLAY.get(r["method"], r["method"])),
                    _fmt(r["lr"], "g") if pd.notna(r["lr"]) else "--",
                    _escape_tex(_SCORE_DISPLAY.get(r["score"], r["score"])),
                    auroc_str,
                    _tex_meanstd(r["aupr_mean"], r["aupr_std"]),
                    _tex_meanstd(r["fpr_at_95_tpr_mean"], r["fpr_at_95_tpr_std"]),
                ]
                lines.append("    " + " & ".join(cells) + r" \\")
            lines.append(r"    \addlinespace")

    lines += [r"\end{longtable}"]

    tex_path = out / f"table_results_{suffix}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tabla guardada en → {tex_path}")


def _bold_if_best(value: str, metric: float, best: float) -> str:
    return rf"\textbf{{{value}}}" if pd.notna(metric) and metric == best else value


def _write_tab_comparison_tex(
    agg: pd.DataFrame, dist: pd.DataFrame, datasets: list[str], out: Path, suffix: str
) -> None:
    n_cols = 6
    lines = [
        r"\begin{longtable}{lllccc}",
        r"\toprule",
        r"Método & LR & Score & AUROC & AUPR & FPR@95\% \\",
        r"\midrule",
    ]
    for i, dataset in enumerate(datasets):
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
                    "score": "--",
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
                    "score": _escape_tex(b["score"]),
                    "auroc": float(b["auroc_mean"]),
                    "aupr": float(b["aupr_mean"]),
                    "fpr": float(b["fpr_at_95_tpr_mean"]),
                }
            )

        best_auroc = max((row["auroc"] for row in rows_to_write), default=float("nan"))
        for row in rows_to_write:
            raw_score = _SCORE_DISPLAY.get(row["score"], row["score"])
            if r"\\" in raw_score:
                parts = [p.strip() for p in raw_score.split(r"\\")]
                parts = [p.upper() if len(p) <= 3 else p.capitalize() for p in parts]
                cleaned_score = f"{parts[0]} ({parts[1]})"
            else:
                cleaned_score = raw_score

            score_cell = _escape_tex(cleaned_score)

            cells = [
                _escape_tex(_METHOD_DISPLAY.get(row["method"], row["method"])),
                _escape_tex(row["lr"]),
                score_cell,
                _bold_if_best(_fmt(row["auroc"], ".4f"), row["auroc"], best_auroc),
                _fmt(row["aupr"], ".4f"),
                _fmt(row["fpr"], ".4f"),
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{longtable}"]
    tex_path = out / f"table_comparison_{suffix}.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_path}")


def _plot_methods_comparison(
    agg: pd.DataFrame,
    dist: pd.DataFrame,
    datasets: list[str],
    out: Path,
    suffix: str,
    group_title: str,
) -> None:
    methods = ["knn", "mahalanobis", "vae", "ddpm"]
    colors = {
        "knn": "gray",
        "mahalanobis": "black",
        "vae": "steelblue",
        "ddpm": "darkorange",
    }

    # Pre-compute global y_min / y_max across all datasets in this group
    all_values: list[float] = []
    for dataset in datasets:
        for meth in methods:
            if meth in DISTANCE_METHODS:
                rows = dist[(dist["dataset"] == dataset) & (dist["method"] == meth)]
                if not rows.empty:
                    all_values.append(float(rows.loc[rows["auroc"].idxmax(), "auroc"]))
            else:
                sub_agg = agg[(agg["dataset"] == dataset) & (agg["method"] == meth)]
                if not sub_agg.empty:
                    all_values.append(float(sub_agg.loc[sub_agg["auroc_mean"].idxmax(), "auroc_mean"]))

    if all_values:
        shared_ymin = max(0.0, min(all_values) - 0.05)
        shared_ymin = min(shared_ymin, 0.75)
        shared_ymax = min(1.0, max(all_values) + 0.03)
    else:
        shared_ymin, shared_ymax = 0.0, 1.0

    textwidth = 5.65
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(textwidth * len(datasets), textwidth * 0.55),
        squeeze=False,
    )

    for ax, dataset in zip(axes[0], datasets, strict=False):
        values = []
        errors = []
        for meth in methods:
            if meth in DISTANCE_METHODS:
                rows = dist[(dist["dataset"] == dataset) & (dist["method"] == meth)]
                if rows.empty:
                    values.append(float("nan"))
                    errors.append(0.0)
                    continue
                best = rows.loc[rows["auroc"].idxmax()]
                values.append(float(best["auroc"]))
                errors.append(0.0)
            else:
                sub_agg = agg[(agg["dataset"] == dataset) & (agg["method"] == meth)]
                if sub_agg.empty:
                    values.append(float("nan"))
                    errors.append(0.0)
                    continue
                b = sub_agg.loc[sub_agg["auroc_mean"].idxmax()]
                values.append(float(b["auroc_mean"]))
                errors.append(float(b["auroc_std"]))

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
        ax.set_xticklabels(methods)
        ax.set_ylim(shared_ymin, shared_ymax)
        ax.set_ylabel("AUROC")
        ax.set_title(dataset.upper(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(
        fig,
        out / f"{suffix}_methods_comparison.png",
        run=None,
        image_key=None,
        artifact_prefix=f"{suffix}-methods-comparison",
        png_dpi=300,
    )
    plt.close(fig)


def run_aggregate_tables(
    csv_path: str = "results/summary/comparison.csv",
    out_dir: str = "results/summary",
) -> None:
    """Tab 10/10b: mean±std per (method, lr, score) across seeds.
    Tab 11/11b: best-per-method comparison across all configs.
    One set of outputs per dataset group (sicap, pathmnist).
    """
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    df = _validate(pd.read_csv(src))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df["t_steps"] = df["experiment_id"].apply(_parse_t_from_id)

    any_group_found = False
    for group_name, candidate_datasets in _DATASET_GROUPS.items():
        key_datasets = [d for d in candidate_datasets if d in df["dataset"].values]
        if not key_datasets:
            continue
        any_group_found = True

        gen_mask = df["method"].isin(GENERATIVE_METHODS) & df["dataset"].isin(key_datasets)
        gen = df[gen_mask].copy()

        dist_df = df[df["method"].isin(DISTANCE_METHODS) & df["dataset"].isin(key_datasets)].copy()
        dist_df["t_steps"] = 10
        dist_df["lr"] = np.nan
        dist_df["score"] = "--"

        all_data = pd.concat([gen, dist_df], ignore_index=True)

        group_keys = ["dataset", "method", "lr", "t_steps", "score"]
        agg = (
            all_data.groupby(group_keys, dropna=False)[["auroc", "aupr", "fpr_at_95_tpr"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

        ds_label = group_name.upper()
        _write_tab_results_tex(
            agg, key_datasets, out, suffix=group_name,
            caption=(
                f"Resultados completos en {ds_label}: AUROC, AUPR y FPR@95\\%"
                r" (media~$\pm$~desv.\ típica). Mejor AUROC por grupo en negrita."
            ),
            label=f"tab:results_{group_name}_full",
        )

        dist = df[df["method"].isin(DISTANCE_METHODS) & df["dataset"].isin(key_datasets)].copy()
        _write_tab_comparison_tex(agg, dist, key_datasets, out, suffix=group_name)
        _plot_methods_comparison(
            agg, dist, key_datasets, out, suffix=group_name, group_title=group_name.upper()
        )

    if not any_group_found:
        print("No key-dataset data found in CSV (need sicap_c1/sicap_c12 or pathmnist_c1/pathmnist_c2).")

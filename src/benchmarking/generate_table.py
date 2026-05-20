from pathlib import Path

import pandas as pd

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

COLUMN_LABELS = {
    "method": "Method",
    "score": "Score",
    "lr": "LR",
    "seed": "Seed",
    "auroc": "AUROC",
    "aupr": "AUPR",
    "fpr_at_95_tpr": "FPR@95%TPR",
    "threshold_at_5_fpr": "Threshold@5%FPR",
    "tpr_at_5_fpr": "TPR@5%FPR",
}

METHOD_ALIASES = {"vae_toy": "vae", "ddpm_toy": "ddpm"}

DATASET_METHODS = {
    "sicap_c1":  ["vae", "ddpm", "knn", "mahalanobis"],
    "sicap_c12": ["vae", "ddpm", "knn", "mahalanobis"],
    "mnist":     ["vae", "ddpm", "knn", "mahalanobis"],
}

GENERATIVE_METHODS = {"vae", "ddpm"}
DISTANCE_METHODS = {"knn", "mahalanobis"}


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


def _build_subtable(subset: pd.DataFrame, title: str) -> str:
    if subset.empty:
        return f"### {title}\n\nNo results."

    models = (
        subset.sort_values(["auroc", "method", "score"], ascending=[False, True, True])
        .head(10)
        .reset_index(drop=True)
    )

    header_labels = list(COLUMN_LABELS.values())
    header = (
        f"### {title}\n\n"
        "| Rank | " + " | ".join(header_labels) + " |\n"
        "|------|" + "|".join("---" for _ in header_labels) + "|"
    )

    def row(rank: str, r: pd.Series) -> str:
        values = [
            r["method"],
            r["score"],
            _fmt(r["lr"], "g"),
            _fmt(r["seed"], ".0f"),
            _fmt(r["auroc"], ".4f"),
            _fmt(r["aupr"], ".4f"),
            _fmt(r["fpr_at_95_tpr"], ".4f"),
            _fmt(r["threshold_at_5_fpr"], ".4f"),
            _fmt(r["tpr_at_5_fpr"], ".4f"),
        ]
        return f"| {rank} | " + " | ".join(values) + " |"

    lines = [header]
    for i, (_, r) in enumerate(models.iterrows(), start=1):
        lines.append(row(str(i), r))

    return "\n".join(lines)


def _build_table(df: pd.DataFrame, dataset: str, allowed_methods: list[str]) -> str:
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        return f"No results for dataset '{dataset}'."

    allowed_lower = [m.lower() for m in allowed_methods]
    subset = subset[subset["method"].isin(allowed_lower)]
    if subset.empty:
        return f"No results for dataset '{dataset}' with methods {allowed_methods}."

    subset = subset.copy()

    generative_subset = subset[subset["method"].isin(GENERATIVE_METHODS)]
    distance_subset = subset[subset["method"].isin(DISTANCE_METHODS)]

    sections = [f"## {dataset.upper()}\n"]
    sections.append(_build_subtable(generative_subset, "Top 10 — Generative models (VAE / DDPM)"))
    sections.append("")
    sections.append(_build_subtable(distance_subset, "Top 10 — Distance methods (KNN / Residual)"))

    return "\n".join(sections)


def run_ranking_tables(
    csv_path: str = "results/summary/comparison.csv",
    out_dir: str = "results/summary",
) -> None:
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    df = _validate(pd.read_csv(src))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for dataset, methods in DATASET_METHODS.items():
        if dataset not in df["dataset"].values:
            print(f"Dataset '{dataset}' no encontrado en el CSV, se omite.")
            continue

        table = _build_table(df, dataset, methods)
        out_path = out / f"table_{dataset}.md"
        out_path.write_text(table, encoding="utf-8")
        print(f"Saved → {out_path}")


def _fmt_meanstd(mean, std, fmt: str = ".4f", fallback: str = "-") -> str:
    if pd.isna(mean):
        return fallback
    s = format(mean, fmt)
    return f"{s} ± {format(std, fmt)}" if not pd.isna(std) and std > 0 else s


def _parse_t_from_id(experiment_id: str) -> int | None:
    import re
    m = re.search(r"_t(\d+)(?:_|$)", str(experiment_id))
    return int(m.group(1)) if m else None


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

    # ── Tab 10: mean ± std ────────────────────────────────────────────────────
    gen_mask = df["method"].isin(GENERATIVE_METHODS) & df["dataset"].isin(sicap_datasets)
    gen = df[gen_mask].copy()

    group_keys = ["dataset", "method", "lr", "t_steps", "score"]
    agg = (
        gen.groupby(group_keys, dropna=False)[["auroc", "aupr", "fpr_at_95_tpr"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "_".join(c).rstrip("_") for c in agg.columns
    ]

    tab10_lines = ["# SICAP — Resultados generativos (media ± std entre semillas)\n"]
    for dataset in sicap_datasets:
        tab10_lines.append(f"## {dataset.upper()}\n")
        sub = agg[agg["dataset"] == dataset].drop(columns="dataset")
        if sub.empty:
            tab10_lines.append("Sin datos.\n")
            continue

        sub = sub.sort_values(["method", "lr", "score", "t_steps"])
        header = "| Método | LR | T | Score | N | AUROC | AUPR | FPR@95 |"
        sep    = "|--------|-----|---|-------|---|-------|------|--------|"
        tab10_lines += [header, sep]

        for _, r in sub.iterrows():
            t_str = str(int(r["t_steps"])) if pd.notna(r["t_steps"]) else "-"
            n     = int(r["auroc_count"]) if pd.notna(r["auroc_count"]) else 1
            tab10_lines.append(
                f"| {r['method']} | {_fmt(r['lr'], 'g')} | {t_str} | {r['score']} | {n}"
                f" | {_fmt_meanstd(r['auroc_mean'], r['auroc_std'])}"
                f" | {_fmt_meanstd(r['aupr_mean'], r['aupr_std'])}"
                f" | {_fmt_meanstd(r['fpr_at_95_tpr_mean'], r['fpr_at_95_tpr_std'])} |"
            )
        tab10_lines.append("")

    tab10_path = out / "table_results_sicap.md"
    tab10_path.write_text("\n".join(tab10_lines), encoding="utf-8")
    print(f"Saved → {tab10_path}")

    # ── Tab 11: best-per-method comparison ───────────────────────────────────
    dist_mask = df["method"].isin(DISTANCE_METHODS) & df["dataset"].isin(sicap_datasets)
    dist = df[dist_mask].copy()

    tab11_lines = ["# SICAP — Comparativa de métodos (mejor configuración)\n"]
    cols = ["Método", "Config", "AUROC", "AUPR", "FPR@95"]
    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join("---" for _ in cols) + "|"

    for dataset in sicap_datasets:
        tab11_lines.append(f"## {dataset.upper()}\n")
        tab11_lines += [header, sep]

        # Distance methods (knn, mahalanobis) — no seed sweep, show as-is
        for meth in ("knn", "mahalanobis"):
            rows = dist[(dist["dataset"] == dataset) & (dist["method"] == meth)]
            if rows.empty:
                continue
            best = rows.loc[rows["auroc"].idxmax()]
            tab11_lines.append(
                f"| {meth} | seed={_fmt(best['seed'], '.0f')} | {_fmt(best['auroc'], '.4f')}"
                f" | {_fmt(best['aupr'], '.4f')} | {_fmt(best['fpr_at_95_tpr'], '.4f')} |"
            )

        # Generative methods — best mean AUROC across seeds
        for meth in ("vae", "ddpm"):
            sub_agg = agg[(agg["dataset"] == dataset) & (agg["method"] == meth)]
            if sub_agg.empty:
                continue
            best_idx = sub_agg["auroc_mean"].idxmax()
            b = sub_agg.loc[best_idx]
            t_str = f", T={int(b['t_steps'])}" if pd.notna(b["t_steps"]) else ""
            cfg = f"lr={_fmt(b['lr'], 'g')}{t_str}, score={b['score']}, N={int(b['auroc_count'])}"
            tab11_lines.append(
                f"| {meth} | {cfg}"
                f" | {_fmt_meanstd(b['auroc_mean'], b['auroc_std'])}"
                f" | {_fmt_meanstd(b['aupr_mean'], b['aupr_std'])}"
                f" | {_fmt_meanstd(b['fpr_at_95_tpr_mean'], b['fpr_at_95_tpr_std'])} |"
            )

        tab11_lines.append("")

    tab11_path = out / "table_comparison_sicap.md"
    tab11_path.write_text("\n".join(tab11_lines), encoding="utf-8")
    print(f"Saved → {tab11_path}")


if __name__ == "__main__":
    run_ranking_tables()

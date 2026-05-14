from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "file",
    "method",
    "dataset",
    "seed",
    "lr",
    "epochs",
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
    "epochs": "Epochs",
    "auroc": "AUROC",
    "aupr": "AUPR",
    "fpr_at_95_tpr": "FPR@95%TPR",
    "threshold_at_5_fpr": "Threshold@5%FPR",
    "tpr_at_5_fpr": "TPR@5%FPR",
}

DATASET_METHODS = {
    "sicap": ["vae", "ddpm", "feature_distance"],
    "mnist": ["vae", "ddpm", "feature_distance"],
}

GENERATIVE_METHODS = {"vae", "ddpm"}
DISTANCE_METHODS = {"feature_distance"}


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"comparison.csv is missing columns: {missing}")
    df = df.copy()
    for col in (
        "lr",
        "seed",
        "epochs",
        "auroc",
        "aupr",
        "fpr_at_95_tpr",
        "threshold_at_5_fpr",
        "tpr_at_5_fpr",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("method", "dataset", "score"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["dataset"] = df["dataset"].replace("data_sicap", "sicap")
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
            _fmt(r["epochs"], ".0f"),
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


def run_generate_table(
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


if __name__ == "__main__":
    run_generate_table()

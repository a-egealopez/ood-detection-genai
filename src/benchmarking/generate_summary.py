import csv
import json
import re
from pathlib import Path


def run_summary(
    results_dir: str = "results", out_csv: str = "results/summary/comparison.csv"
) -> Path:
    root = Path(results_dir)
    out = Path(out_csv)
    rows = []

    for fp in sorted(root.glob("*/eval_results.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            print(f"Could not read {fp}, skipping")
            continue

        metrics_raw = data.get("metrics", {})
        thresholds = data.get("thresholds", {})

        if not metrics_raw:
            print(f"No metrics in {fp}, skipping")
            continue

        dataset = data.get("dataset", "")
        seed = data.get("seed", "")
        lr = data.get("lr", "")
        method = data.get("method", "")

        # Fallback: parse from experiment_id (folder name)
        experiment_id = fp.parent.name
        if not dataset:
            m = re.search(r"_(mnist|sicap_c1|sicap_c12|moons|circles)_", experiment_id)
            if m:
                dataset = m.group(1)
        if not lr:
            m = re.search(r"_lr([0-9e.+-]+)", experiment_id)
            if m:
                lr = m.group(1)
        if not seed:
            m = re.search(r"_s(\d+)_", experiment_id)
            if m:
                seed = m.group(1)
        if not method:
            parts = experiment_id.split("_")
            # dist_knn_... → "knn";  vae_... → "vae";  ddpm_toy_... → "ddpm_toy"
            if parts[0] == "dist" and len(parts) > 1:
                method = parts[1]
            elif len(parts) > 1 and parts[1] == "toy":
                method = f"{parts[0]}_toy"
            else:
                method = parts[0]

        for score, mode_metrics in metrics_raw.items():
            auroc_val = mode_metrics.get("auroc")
            aupr_val = mode_metrics.get("aupr")
            fpr_at_95_tpr_val = mode_metrics.get("fpr_at_95_tpr")

            threshold_val = tpr_at_5_fpr_val = fpr_at_5_fpr_val = None
            if score in thresholds:
                threshold_val = thresholds[score].get("threshold")
                tpr_at_5_fpr_val = thresholds[score].get("tpr")
                fpr_at_5_fpr_val = thresholds[score].get("fpr")

            rows.append(
                {
                    "experiment_id": experiment_id,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "lr": lr,
                    "score": score,
                    "auroc": auroc_val,
                    "aupr": aupr_val,
                    "fpr_at_95_tpr": fpr_at_95_tpr_val,
                    "threshold_at_5_fpr": threshold_val,
                    "tpr_at_5_fpr": tpr_at_5_fpr_val,
                    "fpr_at_5_fpr": fpr_at_5_fpr_val,
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
                "fpr_at_5_fpr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out} ({len(rows)} rows from {len(set(r['experiment_id'] for r in rows))} experiments)")
    return out

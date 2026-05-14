import csv
import json
import re
from pathlib import Path


def run_summary(
    logs_dir: str = "results/logs", out_csv: str = "results/summary/comparison.csv"
) -> Path:
    logs = Path(logs_dir)
    out = Path(out_csv)
    rows = []

    for fp in sorted(logs.rglob("eval_results.json")):
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
        epochs = data.get("epochs", "")
        method = data.get("method", "")

        # Fallback: parse from folder structure results/logs/{method_type}/{experiment_id}/eval_results.json
        parts = fp.parts
        if len(parts) >= 3:
            method_type = parts[-3]
            experiment_id = parts[-2]

            if not dataset:
                m = re.search(r"(mnist|sicap)", experiment_id)
                if m:
                    dataset = m.group(1)
            if not lr:
                m = re.search(r"_lr([0-9e.-]+)", experiment_id)
                if m:
                    lr = m.group(1)
            if not seed:
                m = re.search(r"_s(\d+)", experiment_id)
                if m:
                    seed = m.group(1)
            if not epochs:
                m = re.search(r"_ep(\d+)", experiment_id)
                if m:
                    epochs = m.group(1)
            if not method:
                method = method_type

        for score, mode_metrics in metrics_raw.items():
            auroc_val = mode_metrics.get("auroc")
            aupr_val = mode_metrics.get("aupr")
            fpr_at_95_tpr_val = mode_metrics.get("fpr_at_95_tpr")

            threshold_val = None
            tpr_at_5_fpr_val = None
            fpr_at_5_fpr_val = None
            if score in thresholds:
                threshold_val = thresholds[score].get("threshold")
                tpr_at_5_fpr_val = thresholds[score].get("tpr")
                fpr_at_5_fpr_val = thresholds[score].get("fpr")

            rows.append(
                {
                    "file": str(fp.relative_to(logs)),
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "epochs": epochs,
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
                "fpr_at_5_fpr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out}")
    return out

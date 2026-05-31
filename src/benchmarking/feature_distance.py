from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.artifacts import build_experiment_id
from src.data import build_dataloaders
from src.evaluation.evaluate import compute_metrics
from src.models.ood_scorers import DEFAULT_KNN_K, build_latent_reference, compute_ood_score


def _loader_to_numpy(loader, device: torch.device) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(x.to(device).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _score_features(features: np.ndarray, ref: dict, mode: str, device: torch.device) -> np.ndarray:
    feats = torch.from_numpy(features).float().to(device)
    scores: list[np.ndarray] = []
    for i in range(0, len(feats), 512):
        batch = feats[i : i + 512]
        score = compute_ood_score(feat=batch, mode=mode, reference=ref)
        scores.append(score.cpu().numpy())
    return np.concatenate(scores, axis=0)


def run_feature_distance(cfg, device: torch.device, out: str = "") -> Path:

    loaders = build_dataloaders(cfg)

    x_train = _loader_to_numpy(loaders["train"], device)
    x_id = _loader_to_numpy(loaders["id_eval"], device)
    x_ood = _loader_to_numpy(loaders["ood_eval"], device)

    ref = build_latent_reference(
        x_train,
        knn_k=int(cfg.ood.get("knn_k", DEFAULT_KNN_K)),
        normalize_knn=True,  # deep K-NN: normalize for KNN, raw for Mahalanobis
        reg=float(cfg.ood.get("mahalanobis_reg", 1e-5)),
    )

    current_mode = cfg.distance_type

    metrics = {}
    id_scores = _score_features(x_id, ref, current_mode, device)
    ood_scores = _score_features(x_ood, ref, current_mode, device)

    metrics[current_mode] = compute_metrics(id_scores, ood_scores)
    m = metrics[current_mode]

    print(
        f"  [{current_mode:>12}]  AUROC={m['auroc']:.4f}  AUPR={m['aupr']:.4f}  FPR@95={m['fpr_at_95_tpr']:.4f}"
    )

    result = {
        "method": current_mode,
        "dataset": str(cfg.data.dataset),
        "seed": int(cfg.seed),
        "metrics": metrics,
        "thresholds": {},
    }

    if out:
        out_path = Path(out)
    else:
        experiment_id = build_experiment_id(cfg)
        eval_dir = Path(
            str(cfg.get("evaluation", {}).get("results_dir", Path("results") / experiment_id))
        )
        out_path = eval_dir / "eval_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Results saved → {out_path}")
    return out_path

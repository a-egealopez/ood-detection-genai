from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.artifacts import build_experiment_id
from src.data import build_dataloaders
from src.evaluation.evaluate import compute_metrics
from src.evaluation.extract import DEFAULT_KNN_K, DEFAULT_RESIDUAL_COMPONENTS
from src.models.ood_scorers import build_latent_reference, compute_ood_score


def _loader_to_numpy(loader, device: torch.device) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for x, _ in loader:
            chunks.append(x.to(device).squeeze(1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _score_features(z: np.ndarray, ref: dict, mode: str, device: torch.device) -> np.ndarray:
    z_t = torch.from_numpy(z).float().to(device)
    out = []
    for i in range(0, len(z_t), 512):
        zz = z_t[i : i + 512]
        s = compute_ood_score(
            x=zz.unsqueeze(1),
            z_mu=zz,
            z_logvar=torch.zeros_like(zz),
            mode=mode,
            reference=ref,
        )
        out.append(s.cpu().numpy())
    return np.concatenate(out, axis=0)


def run_feature_distance(cfg, device: torch.device, out: str = "") -> Path:
    """Compute feature-space OOD scores (residual + knn) as a distance baseline.

    Operates directly in pixel/feature space without a generative model.
    Saves a JSON with the same schema as evaluation.py results so that
    generate_summary can process it without special-casing.
    """
    loaders = build_dataloaders(cfg)

    z_train = _loader_to_numpy(loaders["train"], device)
    z_id = _loader_to_numpy(loaders["id_eval"], device)
    z_ood = _loader_to_numpy(loaders["ood_eval"], device)

    ref = build_latent_reference(
        z_train,
        residual_components=int(cfg.ood.get("residual_components", DEFAULT_RESIDUAL_COMPONENTS)),
        knn_k=int(cfg.ood.get("knn_k", DEFAULT_KNN_K)),
    )

    metrics = {}
    for mode in ("residual", "knn"):
        id_scores = _score_features(z_id, ref, mode, device)
        ood_scores = _score_features(z_ood, ref, mode, device)
        metrics[mode] = compute_metrics(id_scores, ood_scores)
        m = metrics[mode]
        print(
            f"  [{mode:>8}]  AUROC={m['auroc']:.4f}  AUPR={m['aupr']:.4f}  FPR@95={m['fpr_at_95_tpr']:.4f}"
        )

    result = {
        "method": "feature_distance",
        "dataset": cfg.data.dataset,
        "seed": int(cfg.seed),
        "metrics": metrics,
        "thresholds": {},
    }

    if out:
        out_path = Path(out)
    else:
        experiment_id = build_experiment_id(cfg)
        out_path = Path("results") / "logs" / "distance" / experiment_id / "eval_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Results saved → {out_path}")
    return out_path

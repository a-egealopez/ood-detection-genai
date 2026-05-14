from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.artifacts import save_json

from .extract import ScoreBundle

_ALL_MODES: list[tuple[str, str, str]] = [
    ("recon", "id_recon", "ood_recon"),
    ("elbo", "id_elbo", "ood_elbo"),
    ("residual", "id_residual", "ood_residual"),
    ("knn", "id_knn", "ood_knn"),
    ("noise", "id_noise", "ood_noise"),
]


def _active_modes(scores: ScoreBundle) -> list[tuple[str, str, str]]:
    return [
        (mode, id_attr, ood_attr)
        for mode, id_attr, ood_attr in _ALL_MODES
        if getattr(scores, id_attr, np.array([])).size > 0
    ]


def _labels(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return y, s


def _auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y, s = _labels(id_scores, ood_scores)
    return float(roc_auc_score(y, s))


def _aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y, s = _labels(id_scores, ood_scores)
    return float(average_precision_score(y, s))


def _fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_target: float = 0.95) -> float:
    y, s = _labels(id_scores, ood_scores)
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, tpr_target)
    return float(fpr[min(idx, len(fpr) - 1)])


def compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": _auroc(id_scores, ood_scores),
        "aupr": _aupr(id_scores, ood_scores),
        "fpr_at_95_tpr": _fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95),
    }


@dataclass
class EvalResults:
    metrics: dict[str, dict[str, float]]
    thresholds: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "thresholds": self.thresholds}


def _threshold_analysis(
    scores: ScoreBundle, active: list[tuple[str, str, str]], fpr_target: float
) -> dict:
    result: dict = {}
    for mode, id_attr, ood_attr in active:
        id_scores, ood_scores = getattr(scores, id_attr), getattr(scores, ood_attr)
        tau = float(np.quantile(id_scores, 1.0 - fpr_target))
        result[mode] = {
            "threshold": tau,
            "fpr": float((id_scores > tau).mean()),
            "tpr": float((ood_scores > tau).mean()),
        }
    return result


def evaluate_ood(
    scores: ScoreBundle,
    cfg: DictConfig,
    run=None,
    fpr_target: float = 0.05,
    experiment_id: str | None = None,
) -> EvalResults:
    """Compute AUROC, AUPR, and FPR@95TPR for all active scoring modes."""
    from src.artifacts import _NoOpRun

    run = run or _NoOpRun()
    active = _active_modes(scores)
    metrics: dict[str, dict[str, float]] = {}
    for mode, id_attr, ood_attr in active:
        m = compute_metrics(getattr(scores, id_attr), getattr(scores, ood_attr))
        metrics[mode] = m
        run.log({f"eval/{k}/{mode}": v for k, v in m.items()})
        print(
            f"  [{mode:>8}]  AUROC={m['auroc']:.4f}  AUPR={m['aupr']:.4f}  FPR@95={m['fpr_at_95_tpr']:.4f}"
        )
    thresholds = _threshold_analysis(scores, active, fpr_target)
    print(f"\n  Decision thresholds (FPR target = {fpr_target:.0%}):")
    print(json.dumps(thresholds, indent=4))
    for mode, d in thresholds.items():
        run.log(
            {
                f"eval/threshold/{mode}": d["threshold"],
                f"eval/tpr/{mode}": d["tpr"],
                f"eval/fpr/{mode}": d["fpr"],
            }
        )
    results = EvalResults(metrics=metrics, thresholds=thresholds)
    experiment_id = experiment_id or cfg.wandb.get("run_name", cfg.model.model_type)
    out_path = (
        Path(cfg.get("logs_dir", "results/logs"))
        / cfg.model.model_type
        / experiment_id
        / "eval_results.json"
    )
    save_json(
        payload=results.to_dict(),
        out_path=out_path,
        run=run if cfg.wandb.enabled else None,
        artifact_type="evaluation",
        artifact_prefix="eval-results",
        metadata={
            "model_type": str(cfg.model.model_type),
            "dataset": str(cfg.data.dataset),
            "experiment_id": str(experiment_id),
        },
    )
    print(f"\n  Results saved → {out_path}")
    return results

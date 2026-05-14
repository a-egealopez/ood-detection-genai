import json
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

import wandb


class _NoOpRun:
    url = "(wandb disabled)"

    def __init__(self):
        self.summary = {}

    def log(self, *_, **__):
        pass

    def finish(self, *_, **__):
        pass

    def log_artifact(self, *_, **__):
        pass


def build_wandb_run(cfg: DictConfig) -> wandb.sdk.wandb_run.Run | _NoOpRun:
    if not cfg.wandb.enabled:
        return _NoOpRun()

    try:
        from omegaconf import OmegaConf

        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return run
    except Exception as e:
        print(f"  wandb.init failed: {e}")
        return _NoOpRun()


def build_experiment_id(cfg) -> str:
    if cfg.method == "distance-method":
        return f"dist_{cfg.distance_type}_{cfg.data.dataset}_s{cfg.seed}"

    return f"recon_{cfg.model.model_type}_{cfg.data.dataset}_s{cfg.seed}_lr{cfg.training.lr}_ep{cfg.training.epochs}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-").lower()


def _log_artifact(run, name: str, artifact_type: str, path: Path, metadata: dict) -> None:
    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def save_figure(
    fig,
    out_path: str | Path,
    run=None,
    image_key: str | None = None,
    artifact_type: str = "plot",
    artifact_prefix: str = "plot",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save figure locally and optionally log image + artifact to W&B."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if run is not None:
        if image_key:
            run.log({image_key: wandb.Image(fig)})
        _log_artifact(
            run, _slug(f"{artifact_prefix}-{out.stem}"), artifact_type, out, metadata or {}
        )
    return out


def save_json(
    payload: dict[str, Any],
    out_path: str | Path,
    run=None,
    artifact_type: str = "evaluation",
    artifact_prefix: str = "eval",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save JSON locally and optionally log as W&B artifact."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=4))
    if run is not None:
        _log_artifact(
            run, _slug(f"{artifact_prefix}-{out.stem}"), artifact_type, out, metadata or {}
        )
    return out

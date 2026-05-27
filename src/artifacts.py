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

    base = f"{cfg.model.model_type}_{cfg.data.dataset}_s{cfg.seed}_lr{cfg.training.lr}"
    if "ddpm" in str(cfg.model.get("model_type", "")):
        base += f"_t{cfg.model.n_score_steps}"
    return base


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
    save_vector_formats: list[str] | None = None,
    png_dpi: int = 300,
) -> Path:
    """Save figure locally and optionally log image + artifact to W&B."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # normalize target paths: ensure a PNG is always created, and optionally
    # create vector formats (PDF) for publication-quality export.
    if save_vector_formats is None:
        save_vector_formats = ["pdf"]

    # decide base png path
    if out.suffix == "":
        out = out.with_suffix(".png")

    out_png = out.with_suffix(".png")
    fig.savefig(out_png, dpi=png_dpi, bbox_inches="tight")

    # export vector formats if requested
    for fmt in save_vector_formats:
        try:
            out_vec = out.with_suffix(f".{fmt}")
            fig.savefig(out_vec, bbox_inches="tight")
        except Exception:
            # non-fatal: continue if a format can't be written
            pass
    if run is not None:
        if image_key:
            run.log({image_key: wandb.Image(fig)})
        # log the PNG as the canonical artifact
        _log_artifact(
            run, _slug(f"{artifact_prefix}-{out_png.stem}"), artifact_type, out_png, metadata or {}
        )
    return out_png


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

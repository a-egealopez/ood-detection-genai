from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.ood_scorers import DEFAULT_KNN_K


@dataclass
class ScoreBundle:
    # VAE without latent / feature-space distance
    id_recon: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_recon: np.ndarray = field(default_factory=lambda: np.array([]))
    id_elbo: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_elbo: np.ndarray = field(default_factory=lambda: np.array([]))
    # VAE latent-space distance
    id_latent_knn: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_latent_knn: np.ndarray = field(default_factory=lambda: np.array([]))
    id_latent_mah: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_latent_mah: np.ndarray = field(default_factory=lambda: np.array([]))
    # DDPM without latent
    id_noise_single: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_noise_single: np.ndarray = field(default_factory=lambda: np.array([]))
    id_noise_multi_mse: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_noise_multi_mse: np.ndarray = field(default_factory=lambda: np.array([]))
    id_noise_multi_cosine: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_noise_multi_cosine: np.ndarray = field(default_factory=lambda: np.array([]))
    # DDPM with residual space
    id_residual_mah: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_residual_mah: np.ndarray = field(default_factory=lambda: np.array([]))
    id_residual_knn: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_residual_knn: np.ndarray = field(default_factory=lambda: np.array([]))


def _fit_ood_reference(
    model, loaders: dict, device: torch.device, cfg: DictConfig | None
) -> dict | None:
    if not hasattr(model, "build_ood_reference"):
        return None
    model.eval()
    return model.build_ood_reference(loaders["train"], device, cfg)


def _score_loader(model, loader: DataLoader, device: torch.device, mode: str) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"  scoring [{mode}]", leave=False):
            scores.append(model.ood_score(x.to(device), mode=mode))
    return np.concatenate(scores)


def compute_all_scores(
    model, loaders: dict, device: torch.device, cfg: DictConfig | None = None
) -> ScoreBundle:
    print(f"Computing OOD scores [{type(model).__name__}]...")

    reference = _fit_ood_reference(model, loaders, device, cfg)
    if hasattr(model, "set_ood_reference"):
        model.set_ood_reference(reference)

    modes = sorted(getattr(model, "SCORE_MODES", {"recon"}))
    scores_by_mode: dict = {}
    for mode in modes:
        scores_by_mode[f"id_{mode}"] = _score_loader(model, loaders["id_eval"], device, mode)
        scores_by_mode[f"ood_{mode}"] = _score_loader(model, loaders["ood_eval"], device, mode)

    bundle = ScoreBundle(**scores_by_mode)

    for mode in modes:
        id_scores = getattr(bundle, f"id_{mode}")
        ood_scores = getattr(bundle, f"ood_{mode}")
        if id_scores.size > 0:
            print(
                f"  [{mode:>20}]  ID  mu={id_scores.mean():.5f}  sigma={id_scores.std():.5f}"
                f"  |  OOD mu={ood_scores.mean():.5f}  sigma={ood_scores.std():.5f}"
            )
    return bundle


def extract_representations(
    model, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "encode"):
        raise AttributeError(f"{type(model).__name__} has no `encode` method.")

    model.eval()
    embeddings, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            z, _ = model.encode(x.to(device))
            embeddings.append(z.cpu().numpy())
            targets.append(y if isinstance(y, np.ndarray) else y.numpy())
    return np.concatenate(embeddings), np.concatenate(targets)

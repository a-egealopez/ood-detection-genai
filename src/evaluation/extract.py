from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEFAULT_KNN_K = 1
DEFAULT_RESIDUAL_COMPONENTS = 10


@dataclass
class ScoreBundle:
    id_recon: np.ndarray
    ood_recon: np.ndarray
    id_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    id_knn: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_knn: np.ndarray = field(default_factory=lambda: np.array([]))
    id_elbo: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_elbo: np.ndarray = field(default_factory=lambda: np.array([]))
    id_noise: np.ndarray = field(default_factory=lambda: np.array([]))
    ood_noise: np.ndarray = field(default_factory=lambda: np.array([]))


def build_latent_reference(latents: np.ndarray, residual_components: int, knn_k: int) -> dict:
    z = torch.as_tensor(latents, dtype=torch.float32)

    z_centered = z - z.mean(dim=0)
    ztz = z_centered.T @ z_centered
    evals, evecs = torch.linalg.eigh(ztz)
    order = torch.argsort(evals, descending=True)
    q = evecs[:, order]

    n_components = int(max(1, min(residual_components, q.shape[1])))
    residual_basis = q[:, -n_components:]

    zn = z / (z.norm(dim=1, keepdim=True) + 1e-12)
    knn = NearestNeighbors(n_neighbors=max(knn_k, 1))
    knn.fit(zn.cpu().numpy())

    return {
        "residual_basis": residual_basis,
        "knn_index": knn,
        "knn_k": max(knn_k, 1),
    }


def _score_loader(model, loader: DataLoader, device: torch.device, mode: str) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"  scoring [{mode}]", leave=False):
            parts.append(model.ood_score(x.to(device), mode=mode))
    return np.concatenate(parts)


def _fit_ood_reference(
    model, loaders: dict, device: torch.device, cfg: DictConfig | None
) -> dict | None:
    if not hasattr(model, "encode"):
        return None

    model.eval()
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in tqdm(loaders["train"], desc="  fitting latent reference", leave=False):
            z, _ = model.encode(x.to(device))
            latents.append(z.detach().cpu().numpy())
    if not latents:
        return None

    z_train = np.concatenate(latents, axis=0)
    residual_components = (
        int(cfg.ood.get("residual_components", DEFAULT_RESIDUAL_COMPONENTS))
        if cfg
        else DEFAULT_RESIDUAL_COMPONENTS
    )
    knn_k = int(cfg.ood.get("knn_k", DEFAULT_KNN_K)) if cfg else DEFAULT_KNN_K
    return build_latent_reference(z_train, residual_components=residual_components, knn_k=knn_k)


def compute_all_scores(
    model, loaders: dict, device: torch.device, cfg: DictConfig | None = None
) -> ScoreBundle:
    print(f"Computing OOD scores [{type(model).__name__}]...")

    reference = _fit_ood_reference(model, loaders, device, cfg)
    if hasattr(model, "set_ood_reference"):
        model.set_ood_reference(reference)

    modes = sorted(getattr(model, "SCORE_MODES", {"recon"}))
    bundle_kwargs: dict = {}
    for mode in modes:
        bundle_kwargs[f"id_{mode}"] = _score_loader(model, loaders["id_eval"], device, mode)
        bundle_kwargs[f"ood_{mode}"] = _score_loader(model, loaders["ood_eval"], device, mode)

    bundle = ScoreBundle(**bundle_kwargs)

    for mode in modes:
        id_scores, ood_scores = getattr(bundle, f"id_{mode}"), getattr(bundle, f"ood_{mode}")
        if id_scores.size > 0:
            print(
                f"  [{mode:>8}]  ID  mu={id_scores.mean():.5f}  sigma={id_scores.std():.5f}"
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

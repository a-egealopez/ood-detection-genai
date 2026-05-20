from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

DEFAULT_KNN_K: int = 1


def compute_ood_score(
    x: torch.Tensor | None = None,
    x_recon: torch.Tensor | None = None,
    feat: torch.Tensor | None = None,
    z_logvar: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    mode: str = "recon",
    recon_fn: nn.Module | None = None,
    reference: dict | None = None,
) -> torch.Tensor:
    if recon_fn is None:
        recon_fn = nn.MSELoss(reduction="none")

    if mode == "recon":
        _assert_not_none(x, x_recon)
        mse_map = recon_fn(x_recon, x)
        return mse_map.reshape(mse_map.shape[0], -1).mean(dim=1)

    if mode == "elbo":
        _assert_not_none(x, x_recon, feat, z_logvar)
        mse_map = recon_fn(x_recon, x)
        recon = mse_map.reshape(mse_map.shape[0], -1).mean(dim=1)
        kl = -0.5 * (1 + z_logvar - feat.pow(2) - z_logvar.exp()).mean(dim=1)
        return recon + kl_weight * kl

    if mode == "mahalanobis":
        _assert_not_none(feat)
        mu = reference["mahalanobis_mean"].to(feat.device)
        cov_inv = reference["mahalanobis_cov_inv"].to(feat.device)
        diff = feat - mu
        return (diff @ cov_inv * diff).sum(dim=1)

    if mode == "knn":
        _assert_not_none(feat)
        if reference is None or "knn_index" not in reference:
            return feat.norm(dim=1)
        normalize = bool(reference.get("knn_normalize", False))
        feat_query = feat / (feat.norm(dim=1, keepdim=True) + 1e-12) if normalize else feat
        dists, _ = reference["knn_index"].kneighbors(
            feat_query.detach().cpu().numpy(),
            n_neighbors=int(reference.get("knn_k", 1)),
        )
        return torch.as_tensor(dists[:, -1], device=feat.device, dtype=feat.dtype)

    raise ValueError(f"Unknown scoring mode '{mode}'")


def build_latent_reference(
    latents: np.ndarray,
    knn_k: int,
    normalize_knn: bool = False,
    reg: float = 1e-5,
) -> dict:
    z = torch.as_tensor(latents, dtype=torch.float32)
    N, D = z.shape

    mu = z.mean(dim=0)
    centered = z - mu
    cov = (centered.T @ centered) / N
    cov_reg = cov + reg * torch.eye(D)
    cov_inv = torch.linalg.inv(cov_reg)

    z_knn = z / (z.norm(dim=1, keepdim=True) + 1e-12) if normalize_knn else z
    knn_index = NearestNeighbors(n_neighbors=max(knn_k, 1))
    knn_index.fit(z_knn.cpu().numpy())

    return {
        "mahalanobis_mean": mu,
        "mahalanobis_cov_inv": cov_inv,
        "knn_index": knn_index,
        "knn_k": max(knn_k, 1),
        "knn_normalize": normalize_knn,
    }


def _assert_not_none(*args):
    if any(a is None for a in args):
        raise ValueError("Missing required inputs for selected OOD mode.")

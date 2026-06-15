from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

DEFAULT_KNN_K: int = 1

# for unet results!
_MAHAL_MAX_DIM = 2048
_MAHAL_PCA_COMPONENTS = 256


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
        # for unet results!
        if reference is not None and "pca_components" in reference:
            pca_mean = reference["pca_mean"].to(feat.device)
            pca_components = reference["pca_components"].to(feat.device)  # (K, D)
            feat = (feat.reshape(feat.shape[0], -1) - pca_mean) @ pca_components.T
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

    # for unet results!
    pca_mean_t: torch.Tensor | None = None
    pca_components_t: torch.Tensor | None = None
    if D > _MAHAL_MAX_DIM:
        n_components = min(_MAHAL_PCA_COMPONENTS, N - 1, D)
        print(
            f"  [build_latent_reference] D={D} > {_MAHAL_MAX_DIM}: PCA({n_components}) for Mahalanobis"
        )
        pca = PCA(n_components=n_components)
        z_pca = torch.as_tensor(pca.fit_transform(z.cpu().numpy()), dtype=torch.float32)
        pca_mean_t = torch.as_tensor(pca.mean_, dtype=torch.float32)
        pca_components_t = torch.as_tensor(pca.components_, dtype=torch.float32)  # (K, D)
    else:
        z_pca = z

    mu = z_pca.mean(dim=0)
    centered = z_pca - mu
    cov = (centered.T @ centered) / N
    cov_reg = cov + reg * torch.eye(z_pca.shape[1])
    cov_inv = torch.linalg.inv(cov_reg)

    z_knn = z / (z.norm(dim=1, keepdim=True) + 1e-12) if normalize_knn else z
    knn_index = NearestNeighbors(n_neighbors=max(knn_k, 1))
    knn_index.fit(z_knn.cpu().numpy())

    ref = {
        "mahalanobis_mean": mu,
        "mahalanobis_cov_inv": cov_inv,
        "knn_index": knn_index,
        "knn_k": max(knn_k, 1),
        "knn_normalize": normalize_knn,
    }
    if pca_mean_t is not None:
        ref["pca_mean"] = pca_mean_t
        ref["pca_components"] = pca_components_t
    return ref


def _assert_not_none(*args):
    if any(a is None for a in args):
        raise ValueError("Missing required inputs for selected OOD mode.")

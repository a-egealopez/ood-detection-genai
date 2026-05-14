import torch
import torch.nn as nn


def compute_ood_score(
    x: torch.Tensor | None = None,
    x_recon: torch.Tensor | None = None,
    z_mu: torch.Tensor | None = None,
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
        raw = recon_fn(x_recon, x)
        return raw.reshape(raw.shape[0], -1).mean(dim=1)

    if mode == "elbo":
        _assert_not_none(x, x_recon, z_mu, z_logvar)
        raw = recon_fn(x_recon, x)
        recon = raw.reshape(raw.shape[0], -1).mean(dim=1)
        kl = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp()).mean(dim=1)
        return recon + kl_weight * kl

    if mode == "residual":
        _assert_not_none(z_mu)
        if reference is not None and "residual_basis" in reference:
            r = reference["residual_basis"].to(z_mu.device)
            z_perp = (z_mu @ r) @ r.T
            return torch.norm(z_perp, dim=1)
        return z_mu.norm(dim=1)

    if mode == "knn":
        _assert_not_none(z_mu)
        if reference is None or "knn_index" not in reference:
            return z_mu.norm(dim=1)
        zn = z_mu / (z_mu.norm(dim=1, keepdim=True) + 1e-12)
        dists, _ = reference["knn_index"].kneighbors(
            zn.detach().cpu().numpy(),
            n_neighbors=int(reference.get("knn_k", 1)),
        )
        return torch.as_tensor(dists[:, -1], device=z_mu.device, dtype=z_mu.dtype)

    raise ValueError(f"Unknown scoring mode '{mode}'")


def _assert_not_none(*args):
    if any(a is None for a in args):
        raise ValueError("Missing required inputs for selected OOD mode.")

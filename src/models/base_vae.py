from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base_model import BaseOODModel
from .ood_scorers import DEFAULT_KNN_K, build_latent_reference, compute_ood_score


class BaseVAEModel(nn.Module, BaseOODModel):
    SCORE_MODES: frozenset[str] = frozenset({"recon", "elbo", "latent_knn", "latent_mah"})
    _SCORER_MODE: dict[str, str] = {"latent_knn": "knn", "latent_mah": "mahalanobis"}

    def __init__(self) -> None:
        super().__init__()
        self.ood_reference: dict | None = None

    @abstractmethod
    def _prepare_ood_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def _flat_latent(self, z_mu: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def snapshot_fig(self, loaders, cfg, device, epoch, epochs): ...

    @abstractmethod
    def training_info(self) -> dict: ...

    def kl_weight_at(self, epoch: int, cfg) -> float:
        kl_max = float(cfg.training.get("kl_weight", self.kl_weight))
        warmup_epochs = int(cfg.training.get("kl_warmup_epochs", cfg.training.epochs // 4))
        return min(1.0, epoch / max(warmup_epochs, 1)) * kl_max

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor, mode: str = "recon") -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(
                f"Unknown scoring mode '{mode}'. Supports: {sorted(self.SCORE_MODES)}"
            )
        x_flat, x_recon_flat, z_flat, z_logvar_flat = self._prepare_ood_inputs(x)
        scores = compute_ood_score(
            x=x_flat,
            x_recon=x_recon_flat,
            feat=z_flat,
            z_logvar=z_logvar_flat,
            kl_weight=self.kl_weight,
            mode=self._SCORER_MODE.get(mode, mode),
            recon_fn=self._recon_fn,
            reference=self.ood_reference,
        )
        return scores.cpu().numpy()

    def set_ood_reference(self, reference: dict | None) -> None:
        self.ood_reference = reference

    @torch.no_grad()
    def build_ood_reference(self, train_loader, device: torch.device, cfg) -> dict:
        knn_k = int(cfg.ood.get("knn_k", DEFAULT_KNN_K)) if cfg else DEFAULT_KNN_K
        self.eval()
        latents: list[np.ndarray] = []
        for x_batch, _ in train_loader:
            z_mu, _ = self.encode(x_batch.to(device))
            latents.append(self._flat_latent(z_mu).detach().cpu().numpy())
        latents_np = np.concatenate(latents, axis=0)
        return build_latent_reference(latents_np, knn_k=knn_k, normalize_knn=False, reg=1e-5)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

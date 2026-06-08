import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL
from omegaconf import DictConfig

from src.evaluation.plot import render_cell

from .base_model import BaseOODModel
from .ood_scorers import DEFAULT_KNN_K, build_latent_reference, compute_ood_score

_IMG_SIZE = 28


class MonaiVAEModel(nn.Module, BaseOODModel):

    SCORE_MODES: frozenset[str] = frozenset({"recon", "elbo", "latent_knn", "latent_mah"})
    _SCORER_MODE: dict[str, str] = {"latent_knn": "knn", "latent_mah": "mahalanobis"}

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        channels: tuple[int, ...],
        num_res_blocks: int,
        kl_weight: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_dim = in_channels * _IMG_SIZE * _IMG_SIZE
        self.kl_weight = kl_weight
        self.ood_reference: dict | None = None

        self.autoencoder = AutoencoderKL(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=tuple(False for _ in channels),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

        n_pool = len(channels) - 1
        self._latent_spatial = _IMG_SIZE // (2 ** n_pool)
        self._latent_flat_dim = latent_channels * self._latent_spatial ** 2

        self._recon_fn = nn.MSELoss(reduction="none")

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.in_channels, _IMG_SIZE, _IMG_SIZE)

    def _kl_loss(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(z_mu.pow(2) + z_sigma.pow(2) - z_sigma.pow(2).log() - 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_img = self._to_image(x)
        x_recon, z_mu, z_sigma = self.autoencoder(x_img)
        return x_recon, z_mu, z_sigma

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_mu, z_sigma = self.autoencoder.encode(self._to_image(x))
        return z_mu, z_sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode_stage_2_outputs(z)

    def _flat_latent(self, z_mu: torch.Tensor) -> torch.Tensor:
        return z_mu.reshape(z_mu.shape[0], -1)

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z_mu: torch.Tensor,
        z_sigma: torch.Tensor,
        kl_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        beta = float(self.kl_weight if kl_weight is None else kl_weight)
        x_img = self._to_image(x)
        recon = F.l1_loss(x_recon, x_img)
        kl = self._kl_loss(z_mu, z_sigma)
        return {"elbo": recon + beta * kl, "recon": recon, "kl": kl}

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        x_recon, z_mu, z_sigma = self.forward(x)
        return self.loss(x, x_recon, z_mu, z_sigma, kl_weight)

    def kl_weight_at(self, epoch: int, cfg) -> float:
        kl_max = float(cfg.training.get("kl_weight", self.kl_weight))
        warmup_epochs = int(cfg.training.get("kl_warmup_epochs", cfg.training.epochs // 4))
        return min(1.0, epoch / max(warmup_epochs, 1)) * kl_max

    def snapshot_fig(
        self,
        loaders: dict,
        cfg,
        device: torch.device,
        epoch: int,
        epochs: int,
    ) -> plt.Figure:
        self.eval()
        x_batch, labels = next(iter(loaders["id_eval"]))
        n_cols = 8
        x_batch = x_batch[:n_cols].to(device)
        is_image = True

        with torch.no_grad():
            x_recon, _, _ = self.forward(x_batch)
            # Flatten back to match render_cell expectations
            x_recon_flat = x_recon.view(x_recon.shape[0], -1)

        fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))

        for i in range(n_cols):
            axes[0, i].set_title(f"#{i + 1} cls={labels[i].item()}", fontsize=7)
            render_cell(axes[0, i], x_batch[i], is_image, "steelblue")
            render_cell(axes[1, i], x_recon_flat[i], is_image, "darkorange")

        axes[0, 0].set_ylabel("ORIGINAL", fontsize=9, fontweight="bold")
        axes[1, 0].set_ylabel("RECON", fontsize=9, fontweight="bold")
        plt.tight_layout()
        self.train()
        return fig

    def training_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "in_channels": self.in_channels,
            "latent_flat_dim": self._latent_flat_dim,
            "kl_weight": self.kl_weight,
            "parameters": n_params,
        }

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor, mode: str = "recon") -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(
                f"Unknown scoring mode '{mode}'. MonaiVAEModel supports: {sorted(self.SCORE_MODES)}"
            )
        x_img = self._to_image(x)
        z_mu, z_sigma = self.autoencoder.encode(x_img)
        x_recon = self.autoencoder.decode_stage_2_outputs(z_mu)

        x_flat = x_img.reshape(x.shape[0], -1)
        x_recon_flat = x_recon.reshape(x.shape[0], -1)
        z_flat = self._flat_latent(z_mu)
        # sigma -> logvar
        z_logvar = (2.0 * z_sigma.clamp(min=1e-8).log()).reshape(x.shape[0], -1)

        scores = compute_ood_score(
            x=x_flat,
            x_recon=x_recon_flat,
            feat=z_flat,
            z_logvar=z_logvar,
            kl_weight=self.kl_weight,
            mode=self._SCORER_MODE.get(mode, mode),
            recon_fn=nn.MSELoss(reduction="none"),
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


def build_monai_vae_model(cfg: DictConfig, device: torch.device) -> MonaiVAEModel:
    m = cfg.model
    in_channels = int(cfg.data.input_dim) // (_IMG_SIZE * _IMG_SIZE)
    channels = tuple(int(c) for c in m.get("channels", [32, 64, 128]))
    model = MonaiVAEModel(
        in_channels=in_channels,
        latent_channels=int(m.get("latent_channels", 4)),
        channels=channels,
        num_res_blocks=int(m.get("num_res_blocks", 1)),
        kl_weight=float(cfg.training.get("kl_weight", 1e-3)),
    ).to(device)
    return model

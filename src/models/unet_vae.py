import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL
from omegaconf import DictConfig

from src.evaluation.plot import render_cell

from .base_vae import BaseVAEModel

class UnetVAEModel(BaseVAEModel):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        channels: tuple[int, ...],
        num_res_blocks: int,
        kl_weight: float,
        img_size: int = 28,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kl_weight = kl_weight

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
        latent_spatial = img_size // (2 ** n_pool)
        self._latent_flat_dim = latent_channels * latent_spatial ** 2
        self._recon_fn = nn.MSELoss(reduction="none")

    def _flat_latent(self, z_mu: torch.Tensor) -> torch.Tensor:
        return z_mu.reshape(z_mu.shape[0], -1)

    def _prepare_ood_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_sigma = self.autoencoder.encode(x)
        x_recon = self.autoencoder.decode_stage_2_outputs(z_mu)
        x_flat = x.reshape(x.shape[0], -1)
        x_recon_flat = x_recon.reshape(x.shape[0], -1)
        z_flat = self._flat_latent(z_mu)
        z_logvar_flat = (2.0 * z_sigma.clamp(min=1e-8).log()).reshape(x.shape[0], -1)
        return x_flat, x_recon_flat, z_flat, z_logvar_flat

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder.encode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.autoencoder(x)

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        x_recon, z_mu, z_sigma = self.autoencoder(x)
        beta = float(self.kl_weight if kl_weight == 0.0 else kl_weight)
        recon = F.l1_loss(x_recon, x)
        kl = 0.5 * torch.mean(z_mu.pow(2) + z_sigma.pow(2) - z_sigma.pow(2).log() - 1)
        return {"elbo": recon + beta * kl, "recon": recon, "kl": kl}

    def snapshot_fig(self, loaders, cfg, device, epoch, epochs) -> plt.Figure:
        self.eval()
        x_batch, labels = next(iter(loaders["id_eval"]))
        n_cols = 8
        x_batch = x_batch[:n_cols].to(device)

        with torch.no_grad():
            x_recon, _, _ = self.autoencoder(x_batch)

        fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))
        is_image = True
        for i in range(n_cols):
            axes[0, i].set_title(f"#{i + 1} cls={labels[i].item()}", fontsize=7)
            render_cell(axes[0, i], x_batch[i], is_image, "steelblue")
            render_cell(axes[1, i], x_recon[i], is_image, "darkorange")

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


def build_unet_vae_model(cfg: DictConfig, device: torch.device) -> UnetVAEModel:
    m = cfg.model
    # data config takes priority (e.g. 224 for pathmnist), falls back to model config or 28
    img_size = int(cfg.data.get("img_size", m.get("img_size", 28)))
    in_channels = int(cfg.data.input_dim) // (img_size * img_size)
    channels = tuple(int(c) for c in m.get("channels", [32, 64, 128]))
    return UnetVAEModel(
        in_channels=in_channels,
        latent_channels=int(m.get("latent_channels", 4)),
        channels=channels,
        num_res_blocks=int(m.get("num_res_blocks", 1)),
        kl_weight=float(cfg.training.get("kl_weight", 1e-3)),
        img_size=img_size,
    ).to(device)

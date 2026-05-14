import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from omegaconf import DictConfig

from src.evaluation.extract import extract_representations
from src.evaluation.plot import plot_embeddings

from .base_model import BaseOODModel
from .ood_scorers import compute_ood_score


class VAEModel(nn.Module, BaseOODModel):
    _IMAGE_HW: int = 32

    def __init__(self, input_dim: int, latent_dim: int, kl_weight: float) -> None:
        super().__init__()
        if input_dim > 1024:
            raise ValueError(f"VAEModel supports input_dim <= 1024, got {input_dim}.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.ood_reference: dict | None = None

        image_dim = self._IMAGE_HW * self._IMAGE_HW

        self.vae = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(32, 64),
            latent_channels=4,
            sample_size=self._IMAGE_HW,
            norm_num_groups=8,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self._IMAGE_HW, self._IMAGE_HW)
            latent_map = self.vae.encode(dummy).latent_dist.mean
            self._z_map_shape = latent_map.shape[1:]
            self._z_map_flat_dim = int(np.prod(self._z_map_shape))

        self._proj_mu = nn.Linear(self._z_map_flat_dim, latent_dim)
        self._proj_logvar = nn.Linear(self._z_map_flat_dim, latent_dim)
        self._unproj = nn.Linear(latent_dim, self._z_map_flat_dim)

        self._recon_fn = nn.MSELoss(reduction="none")
        self._image_dim = image_dim

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.view(b, -1)
        if flat.shape[1] < self._image_dim:
            pad = self._image_dim - flat.shape[1]
            flat = nn.functional.pad(flat, (0, pad), value=0.0)
        return flat.view(b, 1, self._IMAGE_HW, self._IMAGE_HW)

    def _to_vector(self, img: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        return img.view(b, -1)[:, : self.input_dim].view(b, 1, self.input_dim)

    def _encode_to_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_img = self._to_image(x)
        posterior = self.vae.encode(x_img).latent_dist

        flat_mean = posterior.mean.reshape(x.shape[0], -1)
        flat_logvar = posterior.logvar.reshape(x.shape[0], -1)

        z_mu = self._proj_mu(flat_mean)
        z_logvar = torch.clamp(self._proj_logvar(flat_logvar), min=-10.0, max=10.0)

        return posterior, z_mu, z_logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior, z_mu, z_logvar = self._encode_to_stats(x)
        std = (0.5 * z_logvar).exp()
        z_sample = z_mu + torch.randn_like(std) * std
        z_map = self._unproj(z_sample).view(x.shape[0], *self._z_map_shape)
        x_img_recon = self.vae.decode(z_map).sample
        x_recon = self._to_vector(x_img_recon)
        return x_recon, z_mu, z_logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, z_mu, z_logvar = self._encode_to_stats(x)
        return z_mu, z_logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_map = self._unproj(z).view(z.shape[0], *self._z_map_shape)
        x_img = self.vae.decode(z_map).sample
        return self._to_vector(x_img)

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
        kl_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        beta = float(self.kl_weight if kl_weight is None else kl_weight)
        recon = self._recon_fn(
            x_recon.view(x.shape[0], -1),
            x.view(x.shape[0], -1),
        ).mean()
        kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return {"total": recon + beta * kl, "recon": recon, "kl": kl}

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        x_recon, z_mu, z_logvar = self.forward(x)
        return self.loss(x, x_recon, z_mu, z_logvar, kl_weight)

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
        print(f"\n  Latent snapshot (epoch {epoch}/{epochs})")
        zs, labs = extract_representations(self, loaders["id_eval"], device)
        unique = sorted(set(labs.tolist()))
        fig = plot_embeddings(
            zs,
            labs,
            cfg,
            title=f"Latent Space — epoch {epoch}/{epochs}",
            label_map={v: str(v) for v in unique},
        )
        return fig

    def training_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "latent_dim": self.latent_dim,
            "kl_weight": self.kl_weight,
            "parameters": n_params,
        }

    SCORE_MODES: frozenset[str] = frozenset({"recon", "elbo", "knn", "residual"})

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor, mode: str = "recon") -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(
                f"Unknown scoring mode '{mode}'. VAEModel supports: {sorted(self.SCORE_MODES)}"
            )

        if x.dim() == 2:
            x = x.unsqueeze(1)

        _, z_mu, z_logvar = self._encode_to_stats(x)
        z_map = self._unproj(z_mu).view(x.shape[0], *self._z_map_shape)
        x_recon = self._to_vector(self.vae.decode(z_map).sample)

        scores = compute_ood_score(
            x=x,
            x_recon=x_recon,
            z_mu=z_mu,
            z_logvar=z_logvar,
            kl_weight=self.kl_weight,
            mode=mode,
            recon_fn=self._recon_fn,
            reference=self.ood_reference,
        )
        return scores.cpu().numpy()

    def set_ood_reference(self, reference: dict | None) -> None:
        self.ood_reference = reference

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def build_vae_model(cfg: DictConfig, device: torch.device) -> VAEModel:
    model = VAEModel(
        input_dim=int(cfg.data.input_dim),
        latent_dim=int(cfg.model.latent_dim),
        kl_weight=float(cfg.training.kl_weight),
    ).to(device)
    return model

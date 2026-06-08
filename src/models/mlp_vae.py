import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.evaluation.plot import plot_embeddings

from .base_vae import BaseVAEModel


def _build_encoder_body(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for _ in range(depth):
        layers += [nn.Linear(prev, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
        prev = hidden_dim
    return nn.Sequential(*layers)


def _build_decoder(latent_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class MlpVAEModel(BaseVAEModel):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        kl_weight: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self._encoder_body = _build_encoder_body(input_dim, hidden_dim, depth)
        self._proj_mu = nn.Linear(hidden_dim, latent_dim)
        self._proj_logvar = nn.Linear(hidden_dim, latent_dim)
        self._decoder = _build_decoder(latent_dim, hidden_dim, input_dim, depth)
        self._recon_fn = nn.MSELoss(reduction="none")

    def _flat_latent(self, z_mu: torch.Tensor) -> torch.Tensor:
        return z_mu

    def _prepare_ood_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.shape[0], -1)
        z_mu, z_logvar = self._encode_to_stats(x)
        x_recon = self._decoder(z_mu)
        return x_flat, x_recon, z_mu, z_logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_to_stats(x)

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        x_recon, z_mu, z_logvar = self._forward(x)
        return self._loss(x, x_recon, z_mu, z_logvar, kl_weight)

    def snapshot_fig(self, loaders, cfg, device, epoch, epochs) -> plt.Figure:
        print(f"\n  Latent snapshot (epoch {epoch}/{epochs})")
        from src.evaluation.extract import extract_representations
        zs, labs = extract_representations(self, loaders["id_eval"], device)
        unique = sorted(set(labs.tolist()))
        return plot_embeddings(
            zs, labs, cfg,
            title=f"Latent Space — epoch {epoch}/{epochs}",
            label_map={v: str(v) for v in unique},
        )

    def training_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"latent_dim": self.latent_dim, "kl_weight": self.kl_weight, "parameters": n_params}

    def _encode_to_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder_body(x.view(x.shape[0], -1))
        z_mu = self._proj_mu(h)
        z_logvar = torch.clamp(self._proj_logvar(h), min=-10.0, max=10.0)
        return z_mu, z_logvar

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self._encode_to_stats(x)
        std = (0.5 * z_logvar).exp()
        z_sample = z_mu + torch.randn_like(std) * std
        return self._decoder(z_sample), z_mu, z_logvar

    def _loss(self, x, x_recon, z_mu, z_logvar, kl_weight=None):
        beta = float(self.kl_weight if kl_weight is None else kl_weight)
        x_flat = x.view(x.shape[0], -1)
        recon = self._recon_fn(x_recon, x_flat).mean()
        kl = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return {"elbo": recon + beta * kl, "recon": recon, "kl": kl}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)


def build_mlp_vae_model(cfg: DictConfig, device: torch.device) -> MlpVAEModel:
    return MlpVAEModel(
        input_dim=int(cfg.data.input_dim),
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.get("hidden_dim", 256)),
        depth=int(cfg.model.get("depth", 2)),
        kl_weight=float(cfg.training.kl_weight),
    ).to(device)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.evaluation.plot import plot_embeddings

from .base_model import BaseOODModel
from .ood_scorers import DEFAULT_KNN_K, build_latent_reference, compute_ood_score


def _build_encoder_body(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(depth):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    return nn.Sequential(*layers)


def _build_decoder(latent_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class VAEModel(nn.Module, BaseOODModel):
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
        self.ood_reference: dict | None = None

        self._encoder_body = _build_encoder_body(input_dim, hidden_dim, depth)
        self._proj_mu = nn.Linear(hidden_dim, latent_dim)
        self._proj_logvar = nn.Linear(hidden_dim, latent_dim)
        self._decoder = _build_decoder(latent_dim, hidden_dim, input_dim, depth)
        self._recon_fn = nn.MSELoss(reduction="none")

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

    def _encode_to_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder_body(self._flatten(x))
        z_mu = self._proj_mu(h)
        z_logvar = torch.clamp(self._proj_logvar(h), min=-10.0, max=10.0)
        return z_mu, z_logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self._encode_to_stats(x)
        std = (0.5 * z_logvar).exp()
        z_sample = z_mu + torch.randn_like(std) * std
        x_recon = self._decoder(z_sample)
        return x_recon, z_mu, z_logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_to_stats(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
        kl_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        beta = float(self.kl_weight if kl_weight is None else kl_weight)
        recon = self._recon_fn(x_recon, self._flatten(x)).mean()
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
        from src.evaluation.extract import extract_representations

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

    SCORE_MODES: frozenset[str] = frozenset({"recon", "elbo", "latent_knn", "latent_mah"})

    # Maps public score mode names to the internal strings expected by compute_ood_score.
    _SCORER_MODE: dict[str, str] = {"latent_knn": "knn", "latent_mah": "mahalanobis"}

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor, mode: str = "recon") -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(
                f"Unknown scoring mode '{mode}'. VAEModel supports: {sorted(self.SCORE_MODES)}"
            )
        z_mu, z_logvar = self._encode_to_stats(x)
        x_recon = self._decoder(z_mu)
        scores = compute_ood_score(
            x=self._flatten(x),
            x_recon=x_recon,
            feat=z_mu,
            z_logvar=z_logvar,
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
            latents.append(z_mu.detach().cpu().numpy())
        latents_np = np.concatenate(latents, axis=0)
        return build_latent_reference(latents_np, knn_k=knn_k, normalize_knn=False, reg=1e-5)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def build_vae_model(cfg: DictConfig, device: torch.device) -> VAEModel:
    model = VAEModel(
        input_dim=int(cfg.data.input_dim),
        latent_dim=int(cfg.model.latent_dim),
        hidden_dim=int(cfg.model.get("hidden_dim", 256)),
        depth=int(cfg.model.get("depth", 2)),
        kl_weight=float(cfg.training.kl_weight),
    ).to(device)
    return model

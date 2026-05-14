import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from omegaconf import DictConfig

from src.evaluation.plot import render_cell

from .base_model import BaseOODModel
from .ood_scorers import compute_ood_score


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=x.device) * -scale)
        emb = x[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class GroupNorm1d(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8) -> None:
        super().__init__()
        groups = min(num_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.gn = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x.unsqueeze(-1)).squeeze(-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.norm1 = GroupNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = GroupNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = h
        h = h + self.time_proj(t_emb)
        h = self.fc1(self.act(self.norm1(h)))
        h = self.drop(h)
        h = self.fc2(self.act(self.norm2(h)))
        return residual + h


class ResidualMLPDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim, time_emb_dim, dropout=dropout)
                for _ in range(max(depth, 1))
            ]
        )
        self.out_norm = GroupNorm1d(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.act = nn.SiLU()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        t_emb = self.time_mlp(t.float())
        h = self.in_proj(x_flat)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(self.act(self.out_norm(h))).view_as(x)


class DDPMModel(nn.Module, BaseOODModel):
    SCORE_MODES: frozenset[str] = frozenset({"recon", "noise", "knn", "residual"})

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        time_emb_dim: int,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        prediction_type: str,
        n_score_steps: int,
        recon_timestep: int = 50,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.n_score_steps = n_score_steps
        self.recon_timestep = recon_timestep

        self.ood_reference: dict | None = None
        self._reference_t_encode: int = 40

        self.model = ResidualMLPDenoiser(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
        )
        self._recon_fn = nn.MSELoss(reduction="none")

    def _tweedie(self, x_noisy: torch.Tensor, pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alphas_bar = self.scheduler.alphas_cumprod.to(x_noisy.device)
        ab = alphas_bar[t].view(-1, *([1] * (x_noisy.dim() - 1)))
        if self.prediction_type == "epsilon":
            return (x_noisy - (1 - ab).sqrt() * pred) / ab.sqrt()
        if self.prediction_type == "v_prediction":
            return ab.sqrt() * x_noisy - (1 - ab).sqrt() * pred
        return pred  # prediction_type == "sample"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = torch.full((b,), self.recon_timestep, device=x.device, dtype=torch.long)
        x_recon, x_noisy, _ = self.reconstruct_at_t(x, t)
        return x_recon, x_noisy, t

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        t = torch.randint(1, self.num_train_timesteps, (b,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, t)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x, noise, t)
        else:
            target = x

        pred = self.model(x_noisy, t)
        return {"total": self._recon_fn(pred, target).mean()}

    @torch.no_grad()
    def reconstruct_at_t(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        x_noisy = self.scheduler.add_noise(x, noise, t)
        pred = self.model(x_noisy, t)
        x_recon = self._tweedie(x_noisy, pred, t)
        return x_recon, x_noisy, pred

    @torch.no_grad()
    def denoise_trajectory(
        self,
        x: torch.Tensor,
        t_start: int,
        capture_timesteps: list[int],
        noise: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        t_start = int(max(1, min(t_start, self.num_train_timesteps - 1)))
        capture = sorted(
            {int(t) for t in capture_timesteps if 0 <= int(t) <= t_start},
            reverse=True,
        )
        if noise is None:
            noise = torch.randn_like(x)

        t0 = torch.full((x.shape[0],), t_start, device=x.device, dtype=torch.long)
        current = self.scheduler.add_noise(x, noise, t0)
        states: dict[int, torch.Tensor] = {}
        if t_start in capture:
            states[t_start] = current.detach().clone()

        for t in range(t_start, 0, -1):
            t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            pred = self.model(current, t_vec)
            current = self.scheduler.step(pred, t, current).prev_sample
            if (t - 1) in capture:
                states[t - 1] = current.detach().clone()

        return states

    @torch.no_grad()
    def encode(self, x: torch.Tensor, t_encode: int = 40) -> tuple[torch.Tensor, None]:
        t = torch.full((x.shape[0],), t_encode, device=x.device, dtype=torch.long)
        rng = torch.Generator(device=x.device).manual_seed(t_encode)
        noise = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype)
        x_noisy = self.scheduler.add_noise(x, noise, t)
        pred = self.model(x_noisy, t)
        z = self._tweedie(x_noisy, pred, t)
        return z.squeeze(1), None

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        return self.loss(x)

    def kl_weight_at(self, epoch: int, cfg) -> float:
        return 0.0

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
        max_t = self.num_train_timesteps - 1
        t_grid = torch.tensor(
            [1, 25, 50, 100, 250, 500, 750, max_t], device=device, dtype=torch.long
        )

        is_image = (
            bool(cfg.data.get("is_image", False)) and int(cfg.data.get("input_dim", 0)) == 784
        )
        n_rows = 2 if is_image else 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, n_rows * 2.5))
        fig.suptitle(f"DDPM Snapshot — epoch {epoch}/{epochs}", fontsize=12, fontweight="bold")

        with torch.no_grad():
            for i in range(n_cols):
                x_i = x_batch[i : i + 1]
                t_i = t_grid[i : i + 1]
                x_recon, _, _ = self.reconstruct_at_t(x_i, t_i)

                axes[0, i].set_title(
                    f"#{i + 1} cls={labels[i].item()} t={int(t_i.item())}", fontsize=7
                )
                render_cell(axes[0, i], x_i.squeeze(0), is_image, "steelblue")
                render_cell(axes[1, i], x_recon.squeeze(0), is_image, "darkorange")
                if not is_image:
                    err = (x_i.squeeze(0).cpu() - x_recon.squeeze(0).cpu()).abs()
                    render_cell(axes[2, i], err, is_image, "tomato")

        axes[0, 0].set_ylabel("ORIGINAL", fontsize=9, fontweight="bold")
        axes[1, 0].set_ylabel("RECON", fontsize=9, fontweight="bold")
        if not is_image:
            axes[2, 0].set_ylabel("ABS ERROR", fontsize=9, fontweight="bold")

        fig.text(
            0.5,
            0.01,
            "Columns = different ID samples. Header: class and timestep used for corruption.",
            ha="center",
            fontsize=9,
        )
        plt.tight_layout()
        self.train()
        return fig

    def training_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "hidden_dim": self.model.in_proj.out_features,
            "timesteps": self.num_train_timesteps,
            "prediction": self.prediction_type,
            "recon_t": self.recon_timestep,
            "parameters": n_params,
        }

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor, mode: str = "recon") -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(
                f"Unknown scoring mode '{mode}'. DDPMModel supports: {sorted(self.SCORE_MODES)}"
            )

        b = x.shape[0]

        if mode in {"knn", "residual"}:
            z_mu, _ = self.encode(x, t_encode=self._reference_t_encode)
            return (
                compute_ood_score(
                    x=x,
                    z_mu=z_mu,
                    z_logvar=torch.zeros_like(z_mu),
                    mode=mode,
                    reference=self.ood_reference,
                )
                .cpu()
                .numpy()
            )

        ts = torch.linspace(
            1,
            self.num_train_timesteps - 1,
            self.n_score_steps,
            dtype=torch.long,
            device=x.device,
        )
        rng = torch.Generator(device=x.device).manual_seed(0)
        fixed_noise = torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype)
        scores = torch.zeros(b, device=x.device)

        for t_val in ts:
            t = t_val.expand(b)
            x_noisy = self.scheduler.add_noise(x, fixed_noise, t)
            pred = self.model(x_noisy, t)

            if mode == "recon":
                x_recon = self._tweedie(x_noisy, pred, t)
                scores += self._recon_fn(x_recon, x).mean(dim=tuple(range(1, x.dim())))
            else:  # mode == "noise"
                target = (
                    fixed_noise
                    if self.prediction_type == "epsilon"
                    else self.scheduler.get_velocity(x, fixed_noise, t)
                )
                scores += self._recon_fn(pred, target).mean(dim=tuple(range(1, x.dim())))

        return (scores / self.n_score_steps).cpu().numpy()

    def set_ood_reference(self, reference: dict | None) -> None:
        self.ood_reference = reference

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def build_ddpm_model(cfg: DictConfig, device: torch.device) -> DDPMModel:
    m = cfg.model
    return DDPMModel(
        input_dim=int(cfg.data.input_dim),
        hidden_dim=int(m.hidden_dim),
        depth=int(m.depth),
        time_emb_dim=int(m.get("time_emb_dim", 256)),
        num_train_timesteps=int(m.num_train_timesteps),
        beta_start=float(m.beta_start),
        beta_end=float(m.beta_end),
        prediction_type=str(m.prediction_type),
        n_score_steps=int(m.n_score_steps),
        recon_timestep=int(m.get("recon_timestep", 50)),
        dropout=float(m.get("dropout", 0.0)),
    ).to(device)

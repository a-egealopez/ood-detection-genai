import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.evaluation.plot import render_cell

from .base_model import BaseOODModel
from .ood_scorers import DEFAULT_KNN_K, build_latent_reference, compute_ood_score


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


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ff(self.norm(x)) + self.time_proj(t_emb))
        return x + self.drop(h)


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
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim, time_emb_dim, dropout=dropout)
                for _ in range(max(depth, 1))
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.act = nn.GELU()
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
        h = self.act(self.in_proj(x_flat))
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_proj(self.out_norm(h)).view_as(x)


class DDPMModel(nn.Module, BaseOODModel):
    SCORE_MODES: frozenset[str] = frozenset(
        {"noise_single", "noise_multi_mse", "noise_multi_cosine",
         "recon_single", "recon_multi", "residual_mah", "residual_knn"}
    )

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
        recon_timestep: int = 100,
        noise_timestep: int = 100,
        ood_seed: int = 0,
        dropout: float = 0.0,
        clip_sample: bool = True,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.n_score_steps = n_score_steps
        self.recon_timestep = recon_timestep
        self.noise_timestep = noise_timestep
        self.ood_seed = ood_seed

        self.ood_reference: dict | None = None

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
            clip_sample=clip_sample,
            clip_sample_range=1.0,
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
        return {"noise_mse": self._recon_fn(pred, target).mean()}

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
        from_noise: bool = False,
    ) -> dict[int, torch.Tensor]:
        t_start = int(max(1, min(t_start, self.num_train_timesteps - 1)))
        capture = sorted(
            {int(t) for t in capture_timesteps if 0 <= int(t) <= t_start},
            reverse=True,
        )
        if noise is None:
            noise = torch.randn_like(x)

        if from_noise:
            current = noise.clone()
        else:
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
        return z, None

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        return self.loss(x)

    def kl_weight_at(self, epoch: int, cfg) -> float:
        return 0.0

    def _snapshot_fig_2d(
        self, loaders: dict, cfg, device: torch.device, epoch: int, epochs: int
    ) -> plt.Figure:
        """Scatter-plot snapshot for 2D toy data."""
        xs = torch.cat([x for x, _ in loaders["id_eval"]], dim=0).to(device)
        t = torch.full((xs.shape[0],), self.noise_timestep, device=device, dtype=torch.long)
        noise = self._fixed_noise(xs.shape, device)
        with torch.no_grad():
            x_recon, _, _ = self.reconstruct_at_t(xs, t, noise=noise)
        orig = xs.cpu().numpy()
        recon = x_recon.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"DDPM Snapshot (2D) — epoch {epoch}/{epochs}", fontsize=12, fontweight="bold")
        kw = dict(s=10, alpha=0.65)
        axes[0].scatter(orig[:, 0], orig[:, 1], c="steelblue", **kw)
        axes[0].set_title("Original ID")
        axes[0].set_aspect("equal")
        axes[0].grid(True, alpha=0.3)
        axes[1].scatter(recon[:, 0], recon[:, 1], c="darkorange", **kw)
        axes[1].set_title(f"Tweedie recon (t={self.noise_timestep})")
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        self.train()
        return fig

    def snapshot_fig(
        self,
        loaders: dict,
        cfg,
        device: torch.device,
        epoch: int,
        epochs: int,
    ) -> plt.Figure:
        self.eval()
        is_image = (
            bool(cfg.data.get("is_image", False)) and int(cfg.data.get("input_dim", 0)) == 784
        )
        if not is_image and int(cfg.data.get("input_dim", 0)) == 2:
            return self._snapshot_fig_2d(loaders, cfg, device, epoch, epochs)

        x_batch, labels = next(iter(loaders["id_eval"]))
        n_cols = 8
        x_batch = x_batch[:n_cols].to(device)
        max_t = self.num_train_timesteps - 1
        t_grid = torch.tensor(
            [1, 25, 50, 100, 250, 500, 750, max_t], device=device, dtype=torch.long
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

    def _mse_score(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._recon_fn(x_recon, x).mean(dim=tuple(range(1, x.dim())))

    def _cosine_dist(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_vec = x.reshape(x.shape[0], -1)
        xr_vec = x_recon.reshape(x_recon.shape[0], -1)
        cos_sim = (x_vec * xr_vec).sum(dim=1) / (x_vec.norm(dim=1) * xr_vec.norm(dim=1) + 1e-12)
        return 1.0 - cos_sim

    ###
    def _extract_pred_noise(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        x_noisy = self.scheduler.add_noise(x, noise, t)
        pred = self.model(x_noisy, t)
        if self.prediction_type == "epsilon":
            return pred
        elif self.prediction_type == "v_prediction":
            alphas_bar = self.scheduler.alphas_cumprod.to(x.device)
            ab = alphas_bar[t].view(-1, *([1] * (x.dim() - 1)))
            return ab.sqrt() * noise + (1 - ab).sqrt() * pred
        else:
            alphas_bar = self.scheduler.alphas_cumprod.to(x.device)
            ab = alphas_bar[t].view(-1, *([1] * (x.dim() - 1)))
            x_noisy = self.scheduler.add_noise(x, noise, t)
            return (x_noisy - ab.sqrt() * pred) / (1 - ab).sqrt().clamp(min=1e-8)

    def _noise_pred_error(self, x, t, noise):
        pred_noise = self._extract_pred_noise(x, t, noise)
        return self._recon_fn(pred_noise, noise).mean(dim=tuple(range(1, x.dim())))

    def _noise_pred_cosine(self, x, t, noise):
        pred_noise = self._extract_pred_noise(x, t, noise)
        return self._cosine_dist(pred_noise, noise)

    ###

    def _fixed_noise(self, shape: tuple, device: torch.device) -> torch.Tensor:
        rng = torch.Generator(device=device).manual_seed(self.ood_seed)
        return torch.randn(shape, generator=rng, device=device)

    def _score_timesteps(self, device: torch.device) -> torch.Tensor:
        t_min = max(1, int(self.num_train_timesteps * 0.10))
        t_max = int(self.num_train_timesteps * 0.90)
        return torch.linspace(t_min, t_max, self.n_score_steps, dtype=torch.long, device=device)

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor, mode: str = "noise_single") -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(
                f"Unknown scoring mode '{mode}'. DDPMModel supports: {sorted(self.SCORE_MODES)}"
            )

        b = x.shape[0]
        fixed_noise = self._fixed_noise(x.shape, x.device)

        if mode == "noise_single":
            t = torch.full((b,), self.noise_timestep, device=x.device, dtype=torch.long)
            return self._noise_pred_error(x, t, fixed_noise).cpu().numpy()

        if mode == "recon_single":
            traj = self.denoise_trajectory(
                x, t_start=self.recon_timestep, capture_timesteps=[0],
                noise=fixed_noise, from_noise=False,
            )
            return self._mse_score(traj[0], x).cpu().numpy()

        score_timesteps = self._score_timesteps(x.device)

        if mode == "recon_multi":
            if self.ood_reference is None or "recon_multi_stats" not in self.ood_reference:
                raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
            per_level_stats = self.ood_reference["recon_multi_stats"]
            z_scores: list[torch.Tensor] = []
            for t_step in score_timesteps:
                t_key = int(t_step.item())
                traj = self.denoise_trajectory(
                    x, t_start=t_key, capture_timesteps=[0],
                    noise=fixed_noise, from_noise=False,
                )
                recon_mse = self._mse_score(traj[0], x)
                z_scores.append(
                    (recon_mse - per_level_stats[t_key]["mean"]) / per_level_stats[t_key]["std"]
                )
            return torch.stack(z_scores, dim=1).mean(dim=1).cpu().numpy()

        if mode == "noise_multi_mse":
            if self.ood_reference is None or "noise_multi_stats_mse" not in self.ood_reference:
                raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
            per_level_stats = self.ood_reference["noise_multi_stats_mse"]
            z_scores: list[torch.Tensor] = []
            for t_step in score_timesteps:
                t_key = int(t_step.item())
                t_batch = torch.full((b,), t_key, device=x.device, dtype=torch.long)
                x_noisy = self.scheduler.add_noise(x, fixed_noise, t_batch)
                self._tweedie(x_noisy, self.model(x_noisy, t_batch), t_batch)
                mse_score = self._noise_pred_error(x, t_batch, fixed_noise)
                z_scores.append(
                    (mse_score - per_level_stats[t_key]["mean"]) / per_level_stats[t_key]["std"]
                )
            # INVERTIMOS porque COSINE es medida de SIMILITUD no de DISTANCIA
            return torch.stack(z_scores, dim=1).mean(dim=1).cpu().numpy()

        if mode == "noise_multi_cosine":
            if self.ood_reference is None or "noise_multi_stats_cosine" not in self.ood_reference:
                raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
            per_level_stats = self.ood_reference["noise_multi_stats_cosine"]
            z_scores = []
            for t_step in score_timesteps:
                t_key = int(t_step.item())
                t_batch = torch.full((b,), t_key, device=x.device, dtype=torch.long)
                x_noisy = self.scheduler.add_noise(x, fixed_noise, t_batch)
                # x_recon = self._tweedie(x_noisy, self.model(x_noisy, t_batch), t_batch)
                cosine_score = self._noise_pred_cosine(x, t_batch, fixed_noise)
                z_scores.append(
                    (cosine_score - per_level_stats[t_key]["mean"]) / per_level_stats[t_key]["std"]
                )

            return torch.stack(z_scores, dim=1).mean(dim=1).cpu().numpy()

        if self.ood_reference is None or "mahalanobis_mean" not in self.ood_reference:
            raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
        t_residual = torch.full((b,), self.recon_timestep, device=x.device, dtype=torch.long)
        x_at_recon_t = self.scheduler.add_noise(x, fixed_noise, t_residual)
        noise_pred_at_t = self.model(x_at_recon_t, t_residual)
        residual = (noise_pred_at_t - fixed_noise).reshape(b, -1)
        score_mode = "mahalanobis" if mode == "residual_mah" else "knn"
        return (
            compute_ood_score(feat=residual, mode=score_mode, reference=self.ood_reference)
            .cpu()
            .numpy()
        )

    def set_ood_reference(self, reference: dict | None) -> None:
        self.ood_reference = reference

    @torch.no_grad()
    def build_ood_reference(self, train_loader, device: torch.device, cfg) -> dict:
        knn_k = int(cfg.ood.get("knn_k", DEFAULT_KNN_K)) if cfg else DEFAULT_KNN_K
        score_timesteps = self._score_timesteps(device)
        t_keys = [int(t.item()) for t in score_timesteps]

        residuals: list[np.ndarray] = []
        mse_stats_per_t: dict[int, list[float]] = {k: [] for k in t_keys}
        cosine_stats_per_t: dict[int, list[float]] = {k: [] for k in t_keys}
        recon_stats_per_t: dict[int, list[float]] = {k: [] for k in t_keys}
        _MAX_RECON_REF = 512  # cap iterative recon reference (expensive: t denoising steps/sample)
        n_recon_seen = 0

        self.eval()
        for x_batch, _ in tqdm(train_loader, desc="Building OOD reference", leave=False):
            x = x_batch.to(device)
            b = x.shape[0]
            fixed_noise = self._fixed_noise(x.shape, device)

            t_residual = torch.full((b,), self.recon_timestep, device=device, dtype=torch.long)
            x_at_recon_t = self.scheduler.add_noise(x, fixed_noise, t_residual)
            noise_pred_at_t = self.model(x_at_recon_t, t_residual)
            residuals.append((noise_pred_at_t - fixed_noise).reshape(b, -1).cpu().numpy())

            for t_step in score_timesteps:
                t_key = int(t_step.item())
                t_batch = torch.full((b,), t_key, device=device, dtype=torch.long)
                self.scheduler.add_noise(x, fixed_noise, t_batch)
                noise_err = self._noise_pred_error(x, t_batch, fixed_noise)
                cosine_err = self._noise_pred_cosine(x, t_batch, fixed_noise)
                mse_stats_per_t[t_key].extend(noise_err.cpu().tolist())
                cosine_stats_per_t[t_key].extend(cosine_err.cpu().tolist())

            if n_recon_seen < _MAX_RECON_REF:
                cap = min(b, _MAX_RECON_REF - n_recon_seen)
                x_sub = x[:cap]
                noise_sub = fixed_noise[:cap]
                for t_step in score_timesteps:
                    t_key = int(t_step.item())
                    traj = self.denoise_trajectory(
                        x_sub, t_start=t_key, capture_timesteps=[0],
                        noise=noise_sub, from_noise=False,
                    )
                    recon_stats_per_t[t_key].extend(
                        self._mse_score(traj[0], x_sub).cpu().tolist()
                    )
                n_recon_seen += cap

        residuals_np = np.concatenate(residuals, axis=0)
        ref = build_latent_reference(residuals_np, knn_k=knn_k, normalize_knn=False, reg=1e-5)

        def _per_t_stats(stats_dict):
            return {
                t_key: {"mean": float(np.mean(vals)), "std": float(np.std(vals) + 1e-8)}
                for t_key, vals in stats_dict.items()
            }

        ref["noise_multi_stats_mse"]    = _per_t_stats(mse_stats_per_t)
        ref["noise_multi_stats_cosine"] = _per_t_stats(cosine_stats_per_t)
        ref["recon_multi_stats"]        = _per_t_stats(recon_stats_per_t)
        return ref

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
        recon_timestep=int(m.get("recon_timestep", 250)),
        noise_timestep=int(m.get("noise_timestep", 250)),
        ood_seed=int(cfg.get("seed", 0)),
        dropout=float(m.get("dropout", 0.0)),
        clip_sample=bool(m.get("clip_sample", True)),
    ).to(device)

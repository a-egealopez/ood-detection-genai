from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .base_model import BaseOODModel
from .ood_scorers import DEFAULT_KNN_K, build_latent_reference, compute_ood_score


class BaseDDPMModel(nn.Module, BaseOODModel):
    SCORE_MODES: frozenset[str] = frozenset(
        {
            "noise_single",
            "noise_multi_mse",
            "noise_multi_cosine",
            "recon_single",
            "recon_multi",
            "residual_mah",
            "residual_knn",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.ood_reference: dict | None = None

    @abstractmethod
    def _sched_step(self, pred: torch.Tensor, t: int, current: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def snapshot_fig(self, loaders, cfg, device, epoch, epochs): ...

    @abstractmethod
    def training_info(self) -> dict: ...

    def kl_weight_at(self, epoch: int, cfg) -> float:
        return 0.0

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]:
        return self.loss(x)

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
        return {"eps_mse": nn.functional.mse_loss(pred, target)}

    def _tweedie(self, x_noisy: torch.Tensor, pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alphas_bar = self.scheduler.alphas_cumprod.to(x_noisy.device)
        ab = alphas_bar[t].view(-1, *([1] * (x_noisy.dim() - 1)))
        if self.prediction_type == "epsilon":
            return (x_noisy - (1 - ab).sqrt() * pred) / ab.sqrt()
        if self.prediction_type == "v_prediction":
            return ab.sqrt() * x_noisy - (1 - ab).sqrt() * pred
        return pred

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = torch.full((b,), self.recon_timestep, device=x.device, dtype=torch.long)
        x_recon, x_noisy, _ = self.reconstruct_at_t(x, t)
        return x_recon, x_noisy, t

    @torch.no_grad()
    def reconstruct_at_t(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        t_scalar = int(t.max().item())
        x_noisy = self.scheduler.add_noise(x, noise, t)
        states = self.denoise_trajectory(x, t_start=t_scalar, capture_timesteps=[0], noise=noise)
        return states[0], x_noisy, noise

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

        current = (
            noise.clone()
            if from_noise
            else self.scheduler.add_noise(
                x, noise, torch.full((x.shape[0],), t_start, device=x.device, dtype=torch.long)
            )
        )

        states: dict[int, torch.Tensor] = {}
        if t_start in capture:
            states[t_start] = current.detach().clone()

        for t in range(t_start, 0, -1):
            t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            pred = self.model(current, t_vec)
            current = self._sched_step(pred, t, current)
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
        return self._tweedie(x_noisy, pred, t), None

    def _mse_score(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(x_recon, x, reduction="none").mean(
            dim=tuple(range(1, x.dim()))
        )

    def _cosine_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_vec = a.reshape(a.shape[0], -1)
        b_vec = b.reshape(b.shape[0], -1)
        cos_sim = (a_vec * b_vec).sum(1) / (a_vec.norm(dim=1) * b_vec.norm(dim=1) + 1e-12)
        return 1.0 - cos_sim

    def _extract_pred_noise(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        x_noisy = self.scheduler.add_noise(x, noise, t)
        pred = self.model(x_noisy, t)
        if self.prediction_type == "epsilon":
            return pred
        alphas_bar = self.scheduler.alphas_cumprod.to(x.device)
        ab = alphas_bar[t].view(-1, *([1] * (x.dim() - 1)))
        if self.prediction_type == "v_prediction":
            return ab.sqrt() * noise + (1 - ab).sqrt() * pred
        return (x_noisy - ab.sqrt() * pred) / (1 - ab).sqrt().clamp(min=1e-8)

    def _noise_pred_error(self, x, t, noise):
        pred_noise = self._extract_pred_noise(x, t, noise)
        return self._mse_score(pred_noise, noise)

    def _noise_pred_cosine(self, x, t, noise):
        pred_noise = self._extract_pred_noise(x, t, noise)
        return self._cosine_dist(pred_noise, noise)

    def _fixed_noise(self, shape: tuple, device: torch.device, start_idx: int = 0) -> torch.Tensor:
        seed = (self.ood_seed + start_idx) & 0xFFFF_FFFF
        rng = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(shape, generator=rng, device=device)

    def _score_timesteps(self, device: torch.device) -> torch.Tensor:
        t_min = max(1, int(self.num_train_timesteps * 0.10))
        t_max = int(self.num_train_timesteps * 0.90)
        return torch.linspace(t_min, t_max, self.n_score_steps, dtype=torch.long, device=device)

    @torch.no_grad()
    def ood_score(
        self, x: torch.Tensor, mode: str = "noise_single", start_idx: int = 0
    ) -> np.ndarray:
        if mode not in self.SCORE_MODES:
            raise ValueError(f"Unknown scoring mode '{mode}'. Supports: {sorted(self.SCORE_MODES)}")
        b = x.shape[0]
        fixed_noise = self._fixed_noise(x.shape, x.device, start_idx)

        if mode == "noise_single":
            t = torch.full((b,), self.noise_timestep, device=x.device, dtype=torch.long)
            return self._noise_pred_error(x, t, fixed_noise).cpu().numpy()

        if mode == "recon_single":
            traj = self.denoise_trajectory(
                x,
                t_start=self.recon_timestep,
                capture_timesteps=[0],
                noise=fixed_noise,
                from_noise=False,
            )
            return self._mse_score(traj[0], x).cpu().numpy()

        score_timesteps = self._score_timesteps(x.device)

        if mode == "recon_multi":
            if self.ood_reference is None or "recon_multi_stats" not in self.ood_reference:
                raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
            stats = self.ood_reference["recon_multi_stats"]
            z_scores: list[torch.Tensor] = []
            for t_step in score_timesteps:
                t_key = int(t_step.item())
                traj = self.denoise_trajectory(
                    x,
                    t_start=t_key,
                    capture_timesteps=[0],
                    noise=fixed_noise,
                )
                recon_mse = self._mse_score(traj[0], x)
                z_scores.append((recon_mse - stats[t_key]["mean"]) / stats[t_key]["std"])
            return torch.stack(z_scores, dim=1).mean(dim=1).cpu().numpy()

        if mode == "noise_multi_mse":
            if self.ood_reference is None or "noise_multi_stats_mse" not in self.ood_reference:
                raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
            stats = self.ood_reference["noise_multi_stats_mse"]
            z_scores = []
            for t_step in score_timesteps:
                t_key = int(t_step.item())
                t_batch = torch.full((b,), t_key, device=x.device, dtype=torch.long)
                mse = self._noise_pred_error(x, t_batch, fixed_noise)
                z_scores.append((mse - stats[t_key]["mean"]) / stats[t_key]["std"])
            return torch.stack(z_scores, dim=1).mean(dim=1).cpu().numpy()

        if mode == "noise_multi_cosine":
            if self.ood_reference is None or "noise_multi_stats_cosine" not in self.ood_reference:
                raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
            stats = self.ood_reference["noise_multi_stats_cosine"]
            z_scores = []
            for t_step in score_timesteps:
                t_key = int(t_step.item())
                t_batch = torch.full((b,), t_key, device=x.device, dtype=torch.long)
                cosine = self._noise_pred_cosine(x, t_batch, fixed_noise)
                z_scores.append((cosine - stats[t_key]["mean"]) / stats[t_key]["std"])
            return torch.stack(z_scores, dim=1).mean(dim=1).cpu().numpy()

        if self.ood_reference is None or "mahalanobis_mean" not in self.ood_reference:
            raise RuntimeError("OOD reference not built. Call build_ood_reference() first.")
        t_residual = torch.full((b,), self.recon_timestep, device=x.device, dtype=torch.long)
        x_noisy = self.scheduler.add_noise(x, fixed_noise, t_residual)
        noise_pred = self.model(x_noisy, t_residual)
        residual = (noise_pred - fixed_noise).reshape(b, -1)
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

        eval_cfg = cfg.get("evaluation", {}) if cfg else {}
        skip_scores = set(eval_cfg.get("skip_scores", []))
        skip_recon_multi = "recon_multi" in skip_scores
        n_train_ref = int(eval_cfg.get("n_train_ref_samples", float("inf")))

        residuals: list[np.ndarray] = []
        mse_stats_per_t: dict[int, list[float]] = {k: [] for k in t_keys}
        cosine_stats_per_t: dict[int, list[float]] = {k: [] for k in t_keys}
        recon_stats_per_t: dict[int, list[float]] = {k: [] for k in t_keys}
        _MAX_RECON_REF = 128
        n_recon_seen = 0

        self.eval()
        n_seen = 0
        for x_batch, _ in tqdm(train_loader, desc="Building OOD reference", leave=False):
            x = x_batch.to(device)
            b = x.shape[0]
            fixed_noise = self._fixed_noise(x.shape, device, n_seen)

            t_residual = torch.full((b,), self.recon_timestep, device=device, dtype=torch.long)
            x_at_recon_t = self.scheduler.add_noise(x, fixed_noise, t_residual)
            noise_pred_at_t = self.model(x_at_recon_t, t_residual)
            residuals.append((noise_pred_at_t - fixed_noise).reshape(b, -1).cpu().numpy())

            for t_step in score_timesteps:
                t_key = int(t_step.item())
                t_batch = torch.full((b,), t_key, device=device, dtype=torch.long)
                mse_stats_per_t[t_key].extend(
                    self._noise_pred_error(x, t_batch, fixed_noise).cpu().tolist()
                )
                cosine_stats_per_t[t_key].extend(
                    self._noise_pred_cosine(x, t_batch, fixed_noise).cpu().tolist()
                )

            if not skip_recon_multi and n_recon_seen < _MAX_RECON_REF:
                cap = min(b, _MAX_RECON_REF - n_recon_seen)
                x_sub = x[:cap]
                noise_sub = fixed_noise[:cap]
                for t_step in score_timesteps:
                    t_key = int(t_step.item())
                    traj = self.denoise_trajectory(
                        x_sub,
                        t_start=t_key,
                        capture_timesteps=[0],
                        noise=noise_sub,
                    )
                    recon_stats_per_t[t_key].extend(self._mse_score(traj[0], x_sub).cpu().tolist())
                n_recon_seen += cap
            n_seen += b
            if n_seen >= n_train_ref:
                break

        residuals_np = np.concatenate(residuals, axis=0)
        ref = build_latent_reference(residuals_np, knn_k=knn_k, normalize_knn=False, reg=1e-5)

        def _stats(d):
            return {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v) + 1e-8)} for k, v in d.items()
            }

        ref["noise_multi_stats_mse"] = _stats(mse_stats_per_t)
        ref["noise_multi_stats_cosine"] = _stats(cosine_stats_per_t)
        if not skip_recon_multi:
            ref["recon_multi_stats"] = _stats(recon_stats_per_t)
        return ref

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

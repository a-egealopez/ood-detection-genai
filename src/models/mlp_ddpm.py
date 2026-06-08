import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from omegaconf import DictConfig

from src.evaluation.plot import render_cell

from .base_ddpm import BaseDDPMModel


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        half = size // 2
        emb = torch.log(torch.tensor(10000.0)) / max(half - 1, 1)
        emb = torch.exp(-emb * torch.arange(half))
        self.register_buffer("emb", emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = (x * self.scale) * self.emb.unsqueeze(0)
        return torch.cat([e.sin(), e.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, size: int, t_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.ff = nn.Linear(size + t_dim, size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.norm(x), t_emb], dim=-1)
        return x + self.drop(self.act(self.ff(h)))


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._use_fourier = input_dim <= 4

        self.time_emb = SinusoidalEmbedding(time_emb_dim)

        if self._use_fourier:
            self.input_embs = nn.ModuleList(
                [SinusoidalEmbedding(time_emb_dim, scale=25.0) for _ in range(input_dim)]
            )
            concat_size = input_dim * time_emb_dim + time_emb_dim
        else:
            concat_size = input_dim + time_emb_dim

        self.in_proj = nn.Sequential(nn.Linear(concat_size, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_emb_dim, dropout=dropout) for _ in range(max(depth, 1))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        t_emb = self.time_emb(t.float().reshape(-1, 1))
        if self._use_fourier:
            x_enc = torch.cat(
                [self.input_embs[i](x_flat[:, i:i+1]) for i in range(self.input_dim)], dim=-1
            )
        else:
            x_enc = x_flat
        h = self.in_proj(torch.cat([x_enc, t_emb], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(self.out_norm(h)).view_as(x)


class MlpDDPMModel(BaseDDPMModel):
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

        self.model = ResidualMLP(
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

    def _sched_step(self, pred: torch.Tensor, t: int, current: torch.Tensor) -> torch.Tensor:
        return self.scheduler.step(pred, t, current).prev_sample

    def snapshot_fig(self, loaders, cfg, device, epoch, epochs) -> plt.Figure:
        self.eval()
        is_image = bool(cfg.data.get("is_image", False)) and int(cfg.data.get("input_dim", 0)) == 784
        if not is_image and int(cfg.data.get("input_dim", 0)) == 2:
            return self._snapshot_2d(loaders, cfg, device, epoch, epochs)

        x_batch, labels = next(iter(loaders["id_eval"]))
        n_cols = 8
        x_batch = x_batch[:n_cols].to(device)
        max_t = self.num_train_timesteps - 1
        t_grid = torch.tensor([1, 25, 50, 100, 250, 500, 750, max_t], device=device, dtype=torch.long)

        n_rows = 2 if is_image else 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, n_rows * 2.5))

        with torch.no_grad():
            for i in range(n_cols):
                x_i = x_batch[i:i + 1]
                t_i = t_grid[i:i + 1]
                x_recon, _, _ = self.reconstruct_at_t(x_i, t_i)
                axes[0, i].set_title(f"#{i + 1} t={int(t_i.item())}", fontsize=7)
                render_cell(axes[0, i], x_i.squeeze(0), is_image, "steelblue")
                render_cell(axes[1, i], x_recon.squeeze(0), is_image, "darkorange")
                if not is_image:
                    err = (x_i.squeeze(0).cpu() - x_recon.squeeze(0).cpu()).abs()
                    render_cell(axes[2, i], err, is_image, "tomato")

        axes[0, 0].set_ylabel("ORIGINAL", fontsize=9, fontweight="bold")
        axes[1, 0].set_ylabel("RECON", fontsize=9, fontweight="bold")
        if not is_image:
            axes[2, 0].set_ylabel("ABS ERROR", fontsize=9, fontweight="bold")
        plt.tight_layout()
        self.train()
        return fig

    def training_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "hidden_dim": self.model.hidden_dim,
            "timesteps": self.num_train_timesteps,
            "prediction": self.prediction_type,
            "recon_t": self.recon_timestep,
            "parameters": n_params,
        }

    def _snapshot_2d(self, loaders, cfg, device, epoch, epochs) -> plt.Figure:
        xs = torch.cat([x for x, _ in loaders["id_eval"]], dim=0).to(device)
        t = torch.full((xs.shape[0],), self.noise_timestep, device=device, dtype=torch.long)
        noise = self._fixed_noise(xs.shape, device)
        with torch.no_grad():
            x_recon, _, _ = self.reconstruct_at_t(xs, t, noise=noise)
        orig = xs.cpu().numpy()
        recon = x_recon.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        kw = dict(s=10, alpha=0.65)
        axes[0].scatter(orig[:, 0], orig[:, 1], c="steelblue", **kw)
        axes[0].set_title("Original ID")
        axes[0].set_aspect("equal")
        axes[0].grid(True, alpha=0.3)
        axes[1].scatter(recon[:, 0], recon[:, 1], c="darkorange", **kw)
        axes[1].set_title(f"Iterative recon (t={self.noise_timestep})")
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        self.train()
        return fig


def build_mlp_ddpm_model(cfg: DictConfig, device: torch.device) -> MlpDDPMModel:
    m = cfg.model
    return MlpDDPMModel(
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

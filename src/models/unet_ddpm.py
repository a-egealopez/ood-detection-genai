import matplotlib.pyplot as plt
import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from omegaconf import DictConfig

from src.evaluation.plot import render_cell

from .base_ddpm import BaseDDPMModel

_IMG_SIZE = 28


class UnetDDPMModel(BaseDDPMModel):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        attention_levels: tuple[bool, ...],
        num_head_channels: tuple[int, ...],
        num_res_blocks: int,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        prediction_type: str,
        n_score_steps: int,
        recon_timestep: int = 100,
        noise_timestep: int = 100,
        ood_seed: int = 0,
        clip_sample: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.n_score_steps = n_score_steps
        self.recon_timestep = recon_timestep
        self.noise_timestep = noise_timestep
        self.ood_seed = ood_seed

        self.model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule="linear_beta",
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=clip_sample,
        )

    def _sched_step(self, pred: torch.Tensor, t: int, current: torch.Tensor) -> torch.Tensor:
        return self.scheduler.step(pred, t, current)[0]

    def snapshot_fig(self, loaders, cfg, device, epoch, epochs) -> plt.Figure:
        self.eval()
        x_batch, labels = next(iter(loaders["id_eval"]))
        n_cols = 8
        x_batch = x_batch[:n_cols].to(device)
        is_image = True

        t_grid = torch.tensor(
            [1, 25, 50, 100, 250, 500, 750, self.num_train_timesteps - 1],
            device=device, dtype=torch.long,
        )

        fig, axes = plt.subplots(3, n_cols, figsize=(2.2 * n_cols, 7))

        with torch.no_grad():
            for i in range(n_cols):
                x_i = x_batch[i:i + 1]
                t_i = t_grid[i:i + 1]
                x_recon, x_noisy, _ = self.reconstruct_at_t(x_i, t_i)
                axes[0, i].set_title(f"#{i + 1} t={int(t_i.item())}", fontsize=7)
                render_cell(axes[0, i], x_i.squeeze(0), is_image, "steelblue")
                render_cell(axes[1, i], x_noisy.squeeze(0), is_image, "gray")
                render_cell(axes[2, i], x_recon.squeeze(0), is_image, "darkorange")

        axes[0, 0].set_ylabel("ORIGINAL", fontsize=9, fontweight="bold")
        axes[1, 0].set_ylabel("NOISY", fontsize=9, fontweight="bold")
        axes[2, 0].set_ylabel("RECON", fontsize=9, fontweight="bold")
        fig.text(0.5, 0.01, "Columns = different timesteps of corruption.",
                 ha="center", fontsize=9)
        plt.tight_layout()
        self.train()
        return fig

    def training_info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "in_channels": self.in_channels,
            "timesteps": self.num_train_timesteps,
            "prediction": self.prediction_type,
            "recon_t": self.recon_timestep,
            "parameters": n_params,
        }


def build_unet_ddpm_model(cfg: DictConfig, device: torch.device) -> UnetDDPMModel:
    m = cfg.model
    in_channels = int(cfg.data.input_dim) // (_IMG_SIZE * _IMG_SIZE)
    channels = tuple(int(c) for c in m.get("channels", [64, 128, 256]))
    attention_levels = tuple(bool(a) for a in m.get("attention_levels", [False, True, True]))
    num_head_channels = tuple(int(h) for h in m.get("num_head_channels", [0, 64, 128]))
    return UnetDDPMModel(
        in_channels=in_channels,
        channels=channels,
        attention_levels=attention_levels,
        num_head_channels=num_head_channels,
        num_res_blocks=int(m.get("num_res_blocks", 1)),
        num_train_timesteps=int(m.num_train_timesteps),
        beta_start=float(m.beta_start),
        beta_end=float(m.beta_end),
        prediction_type=str(m.prediction_type),
        n_score_steps=int(m.n_score_steps),
        recon_timestep=int(m.get("recon_timestep", 100)),
        noise_timestep=int(m.get("noise_timestep", 100)),
        ood_seed=int(cfg.get("seed", 0)),
        clip_sample=bool(m.get("clip_sample", True)),
    ).to(device)

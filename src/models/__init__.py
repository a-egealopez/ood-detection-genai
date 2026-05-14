from .ddpm import DDPMModel, build_ddpm_model
from .ood_scorers import compute_ood_score
from .vae import VAEModel, build_vae_model

MODEL_REGISTRY: dict = {
    "vae": build_vae_model,
    "ddpm": build_ddpm_model,
}

__all__ = [
    "VAEModel",
    "build_vae_model",
    "DDPMModel",
    "build_ddpm_model",
    "compute_ood_score",
    "MODEL_REGISTRY",
]

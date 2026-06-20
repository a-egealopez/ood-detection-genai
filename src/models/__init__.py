from .mlp_ddpm import MlpDDPMModel, build_mlp_ddpm_model
from .mlp_vae import MlpVAEModel, build_mlp_vae_model
from .ood_scorers import compute_ood_score
from .unet_ddpm import UnetDDPMModel, build_unet_ddpm_model
from .unet_vae import UnetVAEModel, build_unet_vae_model

MODEL_REGISTRY: dict = {
    "mlp_vae": build_mlp_vae_model,
    "mlp_ddpm": build_mlp_ddpm_model,
    "unet_vae": build_unet_vae_model,
    "unet_ddpm": build_unet_ddpm_model,
}

__all__ = [
    "MlpVAEModel",
    "build_mlp_vae_model",
    "MlpDDPMModel",
    "build_mlp_ddpm_model",
    "UnetVAEModel",
    "build_unet_vae_model",
    "UnetDDPMModel",
    "build_unet_ddpm_model",
    "compute_ood_score",
    "MODEL_REGISTRY",
]

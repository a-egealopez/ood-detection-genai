import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def build_config(experiment: str, data: str) -> DictConfig:
    root = Path("configs")
    cfg = OmegaConf.load(root / "base.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "data" / f"{data}.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.load(root / "experiments" / f"{experiment}.yaml"))

    if not cfg.get("experiment_name"):
        raise ValueError("experiment_name is required but missing in config.")

    OmegaConf.resolve(cfg)
    return cfg


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

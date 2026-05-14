from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch


class BaseOODModel(ABC):
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.0) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def kl_weight_at(self, epoch: int, cfg) -> float: ...

    @abstractmethod
    def snapshot_fig(
        self,
        loaders: dict,
        cfg,
        device: torch.device,
        epoch: int,
        epochs: int,
    ) -> plt.Figure: ...

    @abstractmethod
    def training_info(self) -> dict: ...

    @abstractmethod
    def ood_score(self, x: torch.Tensor, mode: str = "recon") -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str, device: torch.device) -> None: ...

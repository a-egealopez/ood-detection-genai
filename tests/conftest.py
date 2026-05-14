import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture
def mock_config_vae():
    cfg_dict = {
        "data": {
            "input_dim": 784,
            "dataset": "mnist",
            "root": "data/MNIST",
            "is_image": True,
        },
        "model": {
            "model_type": "vae",
            "latent_dim": 16,
        },
        "training": {
            "lr": 1e-3,
            "epochs": 5,
            "batch_size": 32,
            "kl_weight": 1.0,
            "kl_warmup_epochs": 1,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "checkpoint_dir": "results/checkpoints",
            "validate_every_n_epochs": 1,
            "viz_every_n_epochs": 5,
            "early_stopping": False,
            "ema_decay": 0.999,
            "use_ema": False,
            "grad_clip": 0.0,
        },
        "wandb": {"enabled": False},
        "seed": 42,
        "experiment_name": "test_vae",
    }
    return OmegaConf.create(cfg_dict)


@pytest.fixture
def mock_config_ddpm():
    cfg_dict = {
        "data": {
            "input_dim": 784,
            "dataset": "mnist",
            "root": "data/MNIST",
            "is_image": True,
        },
        "model": {
            "model_type": "ddpm",
            "hidden_dim": 128,
            "depth": 3,
            "time_emb_dim": 32,
            "num_train_timesteps": 1000,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "prediction_type": "epsilon",
            "n_score_steps": 10,
            "dropout": 0.1,
        },
        "training": {
            "lr": 1e-3,
            "epochs": 5,
            "batch_size": 32,
            "checkpoint_dir": "results/checkpoints",
            "validate_every_n_epochs": 1,
            "viz_every_n_epochs": 5,
            "early_stopping": False,
            "grad_clip": 1.0,
        },
        "wandb": {"enabled": False},
        "seed": 42,
        "experiment_name": "test_ddpm",
    }
    return OmegaConf.create(cfg_dict)


@pytest.fixture
def sample_batch():
    x = torch.randn(4, 1, 784) * 0.5
    y = torch.randint(0, 10, (4,))
    return x, y


@pytest.fixture
def sample_batch_small():
    x = torch.randn(2, 1, 784) * 0.5
    y = torch.randint(0, 10, (2,))
    return x, y


@pytest.fixture
def sample_batch_large():
    x = torch.randn(16, 1, 784) * 0.5
    y = torch.randint(0, 10, (16,))
    return x, y


@pytest.fixture
def mock_dataloader(sample_batch):
    x, y = sample_batch
    x_data = torch.cat([x for _ in range(3)], dim=0)
    y_data = torch.cat([y for _ in range(3)], dim=0)
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def mock_loaders_dict(mock_dataloader):
    return {
        "train": mock_dataloader,
        "id_eval": mock_dataloader,
        "ood_eval": mock_dataloader,
    }


@pytest.fixture
def temp_checkpoint_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "data").mkdir()
        (tmppath / "experiments").mkdir()
        base_cfg = """
seed: 42
wandb:
  enabled: false
  project: test
  tags: []
"""
        (tmppath / "base.yaml").write_text(base_cfg)
        (tmppath / "data" / "mnist.yaml").write_text("data:\n  dataset: mnist\n  input_dim: 784\n")
        (tmppath / "experiments" / "vae.yaml").write_text("model:\n  model_type: vae\n")
        yield tmppath


@pytest.fixture
def mock_wandb_run():
    with mock.patch("wandb.init") as mock_init:
        mock_run = mock.MagicMock()
        mock_run.log = mock.MagicMock()
        mock_run.finish = mock.MagicMock()
        mock_run.log_artifact = mock.MagicMock()
        mock_run.url = "https://wandb.ai/test"
        mock_run.summary = {}
        mock_init.return_value = mock_run
        yield mock_run


@pytest.fixture
def mock_filesystem(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path

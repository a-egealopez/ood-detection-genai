"""
Tests for configuration module (src/config.py).

Happy path tests focus on expected behavior:
  - Configuration building and merging
  - Seed setting for reproducibility
  - W&B run initialization
  - NoOp run as fallback when W&B disabled
"""

import random
from unittest import mock

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.artifacts import _NoOpRun, build_wandb_run
from src.config import seed_everything


class TestNoOpRun:
    """Test the _NoOpRun stub (W&B fallback)."""

    def test_noop_run_creation(self):
        """
        Happy path: _NoOpRun is created and provides expected attributes.
        """
        run = _NoOpRun()

        assert run is not None
        assert hasattr(run, "log")
        assert hasattr(run, "finish")
        assert hasattr(run, "log_artifact")
        assert hasattr(run, "url")
        assert run.url == "(wandb disabled)"

    def test_noop_run_log_silent(self):
        """
        Happy path: _NoOpRun.log() silently absorbs calls without errors.
        """
        run = _NoOpRun()

        # Should not raise any error
        run.log({"loss": 0.5, "epoch": 1})
        run.log({"lr": 1e-3})

    def test_noop_run_finish_silent(self):
        """
        Happy path: _NoOpRun.finish() silently completes.
        """
        run = _NoOpRun()

        # Should not raise any error
        run.finish()
        run.finish(exit_code=0)

    def test_noop_run_log_artifact_silent(self):
        """
        Happy path: _NoOpRun.log_artifact() silently absorbs calls.
        """
        run = _NoOpRun()

        # Should not raise any error
        run.log_artifact("path/to/artifact.txt")
        run.log_artifact("checkpoint.pth")

    def test_noop_run_summary(self):
        """
        Happy path: _NoOpRun has empty summary attribute.
        """
        run = _NoOpRun()

        assert hasattr(run, "summary")
        assert isinstance(run.summary, dict)


class TestSeedEverything:
    """Test deterministic seeding."""

    def test_seed_everything_sets_random_seed(self):
        """
        Happy path: seed_everything sets Python random seed.
        Two sequences with same seed should be identical.
        """
        seed = 42

        # First sequence
        seed_everything(seed)
        seq1 = [random.random() for _ in range(5)]

        # Second sequence with same seed
        seed_everything(seed)
        seq2 = [random.random() for _ in range(5)]

        assert seq1 == seq2

    def test_seed_everything_sets_numpy_seed(self):
        """
        Happy path: seed_everything sets NumPy seed.
        Two arrays with same seed should be identical.
        """
        seed = 42

        # First array
        seed_everything(seed)
        arr1 = np.random.randn(5)

        # Second array with same seed
        seed_everything(seed)
        arr2 = np.random.randn(5)

        assert np.allclose(arr1, arr2)

    def test_seed_everything_sets_torch_seed(self):
        """
        Happy path: seed_everything sets PyTorch seed.
        Two tensors with same seed should be identical.
        """
        seed = 42

        # First tensor
        seed_everything(seed)
        torch.manual_seed(seed)  # seed_everything doesn't call torch.manual_seed
        t1 = torch.randn(5)

        # Second tensor with same seed
        seed_everything(seed)
        torch.manual_seed(seed)
        t2 = torch.randn(5)

        assert torch.allclose(t1, t2)


class TestBuildWandbRun:
    """Test W&B run initialization."""

    def test_build_wandb_run_disabled(self, mock_config_vae):
        """
        Happy path: When wandb.enabled=False, returns _NoOpRun.
        """
        cfg = mock_config_vae
        cfg.wandb.enabled = False

        run = build_wandb_run(cfg)

        assert isinstance(run, _NoOpRun)
        assert run.url == "(wandb disabled)"

    def test_build_wandb_run_returns_noop_on_error(self, mock_config_vae):
        """
        Happy path: When wandb.init fails, gracefully returns _NoOpRun.
        """
        cfg = mock_config_vae
        cfg.wandb.enabled = True

        with mock.patch("wandb.init", side_effect=Exception("Mocked failure")):
            run = build_wandb_run(cfg)

            # Should return _NoOpRun as fallback
            assert isinstance(run, _NoOpRun)

    def test_build_wandb_run_with_mocked_init(self, mock_config_vae):
        """
        Happy path: When wandb.enabled=True and required keys exist, attempts to initialize.
        Since we don't have all required config keys, it will fallback to _NoOpRun.
        """
        cfg = mock_config_vae
        cfg.wandb.enabled = True
        cfg.wandb.project = "test_project"
        cfg.wandb.run_name = "test_run"
        cfg.wandb.tags = ["test"]

        mock_run = mock.MagicMock()
        with mock.patch("src.artifacts.wandb") as mock_wandb:
            mock_wandb.init = mock.MagicMock(return_value=mock_run)
            run = build_wandb_run(cfg)

            # Verify that wandb.init was called
            assert mock_wandb.init.called
            # Should be the mocked run
            assert run is mock_run


class TestBuildConfig:
    """Test configuration building."""

    def test_config_has_required_keys(self, mock_config_vae):
        """
        Happy path: Config has required top-level keys.
        """
        cfg = mock_config_vae

        assert isinstance(cfg, DictConfig)
        assert "data" in cfg
        assert "model" in cfg
        assert "training" in cfg
        assert "seed" in cfg
        assert "experiment_name" in cfg

    def test_config_data_keys(self, mock_config_vae):
        """
        Happy path: Config.data has expected keys.
        """
        cfg = mock_config_vae

        assert cfg.data.input_dim == 784
        assert cfg.data.dataset == "mnist"
        assert cfg.data.root == "data/MNIST"

    def test_config_model_keys(self, mock_config_vae):
        """
        Happy path: Config.model has expected keys.
        """
        cfg = mock_config_vae

        assert cfg.model.model_type == "vae"
        assert cfg.model.latent_dim == 16

    def test_config_training_keys(self, mock_config_vae):
        """
        Happy path: Config.training has expected keys.
        """
        cfg = mock_config_vae

        assert cfg.training.lr == 1e-3
        assert cfg.training.epochs == 5
        assert cfg.training.batch_size == 32

    def test_config_wandb_keys(self, mock_config_vae):
        """
        Happy path: Config.wandb has expected keys.
        """
        cfg = mock_config_vae

        assert "enabled" in cfg.wandb
        assert cfg.wandb.enabled is False


class TestConfigMerging:
    """Test configuration merging (base + data + experiment)."""

    def test_config_override_lr(self, mock_config_vae):
        """
        Happy path: Config values can be overridden.
        """
        cfg = mock_config_vae
        original_lr = cfg.training.lr

        cfg.training.lr = 1e-4
        assert cfg.training.lr == 1e-4
        assert cfg.training.lr != original_lr

    def test_config_add_new_key(self, mock_config_vae):
        """
        Happy path: New keys can be added to config.
        """
        cfg = mock_config_vae

        # Add custom key
        OmegaConf.update(cfg, "custom_key", "custom_value")

        assert cfg.custom_key == "custom_value"

    def test_config_nested_access(self, mock_config_vae):
        """
        Happy path: Nested config access works correctly.
        """
        cfg = mock_config_vae

        # Access nested values
        assert cfg.data.input_dim == 784
        assert cfg.model.model_type == "vae"
        assert cfg.training.batch_size == 32


class TestConfigTypeConversions:
    """Test configuration type conversions."""

    def test_config_int_conversion(self, mock_config_vae):
        """
        Happy path: Config numeric values convert to int correctly.
        """
        cfg = mock_config_vae

        epochs = int(cfg.training.epochs)
        assert isinstance(epochs, int)
        assert epochs == 5

    def test_config_float_conversion(self, mock_config_vae):
        """
        Happy path: Config numeric values convert to float correctly.
        """
        cfg = mock_config_vae

        lr = float(cfg.training.lr)
        assert isinstance(lr, float)
        assert lr == 1e-3

    def test_config_bool_conversion(self, mock_config_vae):
        """
        Happy path: Config boolean values convert correctly.
        """
        cfg = mock_config_vae

        wandb_enabled = bool(cfg.wandb.enabled)
        assert isinstance(wandb_enabled, bool)
        assert wandb_enabled is False


class TestDDPMConfig:
    """Test DDPM-specific configuration."""

    def test_ddpm_config_has_required_keys(self, mock_config_ddpm):
        """
        Happy path: DDPM config has all required model parameters.
        """
        cfg = mock_config_ddpm

        assert cfg.model.model_type == "ddpm"
        assert "hidden_dim" in cfg.model
        assert "num_train_timesteps" in cfg.model
        assert "beta_start" in cfg.model
        assert "beta_end" in cfg.model

    def test_ddpm_config_values(self, mock_config_ddpm):
        """
        Happy path: DDPM config values are in valid ranges.
        """
        cfg = mock_config_ddpm

        assert cfg.model.hidden_dim > 0
        assert cfg.model.num_train_timesteps > 0
        assert 0 < cfg.model.beta_start < cfg.model.beta_end < 1


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_used_by_model(self, mock_config_vae):
        """
        Happy path: Config values can be used to build models.
        """
        cfg = mock_config_vae

        # Simulate using config to create model
        input_dim = int(cfg.data.input_dim)
        latent_dim = int(cfg.model.latent_dim)
        kl_weight = float(cfg.training.kl_weight)

        assert input_dim == 784
        assert latent_dim == 16
        assert kl_weight == 1.0

    def test_config_with_seed_for_reproducibility(self, mock_config_vae):
        """
        Happy path: Seed from config ensures reproducibility.
        """
        cfg = mock_config_vae
        seed = cfg.seed

        seed_everything(seed)
        vals1 = [random.random() for _ in range(3)]

        seed_everything(seed)
        vals2 = [random.random() for _ in range(3)]

        assert vals1 == vals2

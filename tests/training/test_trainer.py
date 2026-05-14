"""
Tests for training module (src/training/trainer.py).

Tested against the real trainer implementation:
  - EMA initialization, update, and copy_to
  - _is_vae model type detection
  - _build_optimizer with Adam and AdamW
  - _build_scheduler with none and cosine variants
  - _validation_loss for VAE and DDPM modes
  - _save_checkpoint persistence and content
  - train_model full loop with VAEModel
"""

from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.config import _NoOpRun
from src.models.vae import VAEModel
from src.training.trainer import (
    EMA,
    _build_optimizer,
    _build_scheduler,
    _is_vae,
    _save_checkpoint,
    _validation_loss,
    train_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vae(input_dim: int = 16, latent_dim: int = 4) -> VAEModel:
    """Small VAEModel that runs fast in tests."""
    return VAEModel(input_dim=input_dim, latent_dim=latent_dim, kl_weight=1.0)


def _make_loader(batch_size: int = 4, input_dim: int = 16, n_batches: int = 2) -> DataLoader:
    """DataLoader with synthetic (B, 1, input_dim) tensors — matches VAEModel input shape."""
    x = torch.randn(batch_size * n_batches, 1, input_dim)
    y = torch.zeros(batch_size * n_batches, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _make_loaders(input_dim: int = 16) -> dict:
    loader = _make_loader(input_dim=input_dim)
    return {"train": loader, "id_eval": loader, "ood_eval": loader}


# ---------------------------------------------------------------------------
# TestEMA
# ---------------------------------------------------------------------------


class TestEMA:
    """Test Exponential Moving Average."""

    def test_ema_initialization(self, device):
        """EMA shadow keys match floating-point params of the model."""
        model = nn.Linear(10, 5).to(device)
        ema = EMA(model, decay=0.999)

        assert ema.decay == 0.999
        assert len(ema.shadow) > 0
        expected_keys = {k for k, v in model.state_dict().items() if v.is_floating_point()}
        assert set(ema.shadow.keys()) == expected_keys

    def test_ema_shadow_initialized_to_model_values(self, device):
        """Shadow values equal model state at creation time."""
        model = nn.Linear(10, 5).to(device)
        ema = EMA(model, decay=0.999)

        for key, shadow_val in ema.shadow.items():
            assert torch.allclose(shadow_val, model.state_dict()[key])

    def test_ema_update_moves_shadow(self, device):
        """After changing model weights and calling update(), shadow changes but
        does not jump all the way to the new model values (decay < 1)."""
        model = nn.Linear(10, 5).to(device)
        ema = EMA(model, decay=0.9)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Set all model params to a fixed value far from initial
        for param in model.parameters():
            param.data.fill_(99.0)

        ema.update(model)

        for key in ema.shadow:
            # Shadow changed from initial
            assert not torch.allclose(ema.shadow[key], initial_shadow[key])
            # Shadow did NOT jump to the new model value (99.0)
            assert not torch.allclose(ema.shadow[key], model.state_dict()[key])

    def test_ema_update_formula(self, device):
        """shadow = decay * shadow + (1 - decay) * param."""
        model = nn.Linear(1, 1, bias=False).to(device)
        # Set weight to 0 so initial shadow = 0
        model.weight.data.fill_(0.0)
        decay = 0.9
        ema = EMA(model, decay=decay)

        # Now set weight to 10
        model.weight.data.fill_(10.0)
        ema.update(model)

        expected = decay * 0.0 + (1 - decay) * 10.0  # = 1.0
        actual = ema.shadow["weight"].item()
        assert abs(actual - expected) < 1e-5

    def test_ema_copy_to_transfers_shadow_to_model(self, device):
        """copy_to() overwrites each model parameter with the shadow value."""
        model = nn.Linear(10, 5).to(device)
        ema = EMA(model, decay=0.999)

        # Fill shadow with 42.0, respecting each tensor's shape
        for key in ema.shadow:
            ema.shadow[key] = torch.full_like(ema.shadow[key], 42.0)

        ema.copy_to(model)

        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert torch.allclose(param.data, torch.full_like(param.data, 42.0), atol=1e-5)


# ---------------------------------------------------------------------------
# TestIsVAE
# ---------------------------------------------------------------------------


class TestIsVAE:
    def test_is_vae_true_for_vae_config(self, mock_config_vae):
        assert _is_vae(mock_config_vae) is True

    def test_is_vae_false_for_ddpm_config(self, mock_config_ddpm):
        assert _is_vae(mock_config_ddpm) is False


# ---------------------------------------------------------------------------
# TestBuildOptimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_adam_is_default(self, mock_config_vae):
        model = nn.Linear(10, 5)
        opt = _build_optimizer(mock_config_vae, model)
        assert isinstance(opt, optim.Adam)

    def test_adam_uses_lr_from_config(self, mock_config_vae):
        mock_config_vae.training.lr = 3e-4
        model = nn.Linear(10, 5)
        opt = _build_optimizer(mock_config_vae, model)
        assert abs(opt.defaults["lr"] - 3e-4) < 1e-10

    def test_adamw_selected_by_name(self, mock_config_vae):
        mock_config_vae.training.optimizer = "adamw"
        mock_config_vae.training.weight_decay = 1e-4
        model = nn.Linear(10, 5)
        opt = _build_optimizer(mock_config_vae, model)
        assert isinstance(opt, optim.AdamW)
        assert abs(opt.defaults["weight_decay"] - 1e-4) < 1e-10

    def test_weight_decay_zero_by_default(self, mock_config_vae):
        model = nn.Linear(10, 5)
        opt = _build_optimizer(mock_config_vae, model)
        assert opt.defaults["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# TestBuildScheduler
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    def test_returns_none_when_scheduler_is_none(self, mock_config_vae):
        mock_config_vae.training.scheduler = "none"
        model = nn.Linear(10, 5)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        assert _build_scheduler(mock_config_vae, opt) is None

    def test_returns_none_when_key_absent(self, mock_config_vae):
        # mock_config_vae has no 'scheduler' key → defaults to "none"
        model = nn.Linear(10, 5)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        assert _build_scheduler(mock_config_vae, opt) is None

    def test_cosine_scheduler_returned(self, mock_config_vae):
        mock_config_vae.training.scheduler = "cosine"
        mock_config_vae.training.epochs = 10
        mock_config_vae.training.warmup_epochs = 0
        model = nn.Linear(10, 5)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = _build_scheduler(mock_config_vae, opt)
        assert scheduler is not None

    def test_cosine_with_warmup_returned(self, mock_config_vae):
        mock_config_vae.training.scheduler = "cosine"
        mock_config_vae.training.epochs = 10
        mock_config_vae.training.warmup_epochs = 2
        model = nn.Linear(10, 5)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = _build_scheduler(mock_config_vae, opt)
        assert scheduler is not None

    def test_cosine_scheduler_steps_without_error(self, mock_config_vae):
        """Scheduler can be stepped multiple times without crashing."""
        mock_config_vae.training.scheduler = "cosine"
        mock_config_vae.training.epochs = 5
        mock_config_vae.training.warmup_epochs = 0
        model = nn.Linear(10, 5)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = _build_scheduler(mock_config_vae, opt)
        for _ in range(5):
            scheduler.step()


# ---------------------------------------------------------------------------
# TestValidationLoss
# ---------------------------------------------------------------------------


class TestValidationLoss:
    """_validation_loss iterates a DataLoader and averages model.loss['total']."""

    def test_vae_mode_returns_non_negative_float(self, device):
        """VAE validation loss is a non-negative float scalar."""
        model = _make_vae().to(device)
        model.eval()
        loader = _make_loader()

        val = _validation_loss(model, loader, device, vae_mode=True, kl_weight=1.0)

        assert isinstance(val, float)
        assert val >= 0
        assert not np.isnan(val)
        assert not np.isinf(val)

    def test_ddpm_mode_returns_non_negative_float(self, device):
        """DDPM validation loss: model.loss(x) is called (no forward needed)."""
        from src.models.ddpm import DDPMModel

        model = DDPMModel(
            input_dim=16,
            hidden_dim=32,
            depth=2,
            time_emb_dim=16,
            num_train_timesteps=100,
            beta_start=1e-4,
            beta_end=0.02,
            prediction_type="epsilon",
            n_score_steps=5,
        ).to(device)
        model.eval()
        loader = _make_loader(input_dim=16)

        val = _validation_loss(model, loader, device, vae_mode=False, kl_weight=0.0)

        assert isinstance(val, float)
        assert val >= 0

    def test_empty_loader_returns_inf(self, device):
        """If the loader yields no batches, result is inf (not a crash)."""
        model = _make_vae().to(device)
        empty_loader = DataLoader(
            TensorDataset(torch.empty(0, 1, 16), torch.empty(0, dtype=torch.long)), batch_size=4
        )

        val = _validation_loss(model, empty_loader, device, vae_mode=True, kl_weight=1.0)

        assert val == float("inf")

    def test_kl_weight_zero_gives_lower_or_equal_loss(self, device):
        """With kl_weight=0, the KL term is suppressed so loss <= loss at kl_weight=1."""
        model = _make_vae().to(device)
        loader = _make_loader()

        val_kl1 = _validation_loss(model, loader, device, vae_mode=True, kl_weight=1.0)
        val_kl0 = _validation_loss(model, loader, device, vae_mode=True, kl_weight=0.0)

        # kl_weight=0 removes KL contribution → loss must be ≤
        assert val_kl0 <= val_kl1 + 1e-6  # small tolerance for float noise


# ---------------------------------------------------------------------------
# TestSaveCheckpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_creates_file_with_correct_name(self, device, temp_checkpoint_dir, mock_config_vae):
        mock_config_vae.training.checkpoint_dir = str(temp_checkpoint_dir)
        model = nn.Linear(10, 5).to(device)
        opt = optim.Adam(model.parameters())

        _save_checkpoint(model, opt, epoch=3, cfg=mock_config_vae)

        assert (temp_checkpoint_dir / "epoch_003.pth").exists()

    def test_checkpoint_has_required_keys(self, device, temp_checkpoint_dir, mock_config_vae):
        mock_config_vae.training.checkpoint_dir = str(temp_checkpoint_dir)
        model = nn.Linear(10, 5).to(device)
        opt = optim.Adam(model.parameters())

        _save_checkpoint(model, opt, epoch=7, cfg=mock_config_vae)

        ckpt = torch.load(temp_checkpoint_dir / "epoch_007.pth", map_location="cpu")
        assert "epoch" in ckpt
        assert "model_state" in ckpt
        assert "optimizer_state" in ckpt
        assert "cfg" in ckpt
        assert ckpt["epoch"] == 7

    def test_checkpoint_epoch_value_matches(self, device, temp_checkpoint_dir, mock_config_vae):
        mock_config_vae.training.checkpoint_dir = str(temp_checkpoint_dir)
        model = nn.Linear(10, 5).to(device)
        opt = optim.Adam(model.parameters())

        _save_checkpoint(model, opt, epoch=42, cfg=mock_config_vae)

        ckpt = torch.load(temp_checkpoint_dir / "epoch_042.pth", map_location="cpu")
        assert ckpt["epoch"] == 42

    def test_checkpoint_model_state_can_be_restored(
        self, device, temp_checkpoint_dir, mock_config_vae
    ):
        """State dict saved and loaded back produces identical parameters."""
        mock_config_vae.training.checkpoint_dir = str(temp_checkpoint_dir)
        model = nn.Linear(10, 5).to(device)
        opt = optim.Adam(model.parameters())

        _save_checkpoint(model, opt, epoch=1, cfg=mock_config_vae)

        ckpt = torch.load(temp_checkpoint_dir / "epoch_001.pth", map_location="cpu")
        restored = nn.Linear(10, 5)
        restored.load_state_dict(ckpt["model_state"])

        for p_orig, p_restored in zip(model.parameters(), restored.parameters(), strict=False):
            assert torch.allclose(p_orig.cpu(), p_restored)

    def test_checkpoint_creates_directory_if_missing(self, temp_checkpoint_dir, mock_config_vae):
        """_save_checkpoint creates the directory via os.makedirs."""
        nested = temp_checkpoint_dir / "a" / "b" / "c"
        mock_config_vae.training.checkpoint_dir = str(nested)
        model = nn.Linear(2, 2)
        opt = optim.Adam(model.parameters())

        _save_checkpoint(model, opt, epoch=1, cfg=mock_config_vae)

        assert (nested / "epoch_001.pth").exists()


# ---------------------------------------------------------------------------
# TestTrainModel
# ---------------------------------------------------------------------------


class TestTrainModel:
    """train_model with a real (small) VAEModel and synthetic loaders.

    We use input_dim=16 and latent_dim=4 so epochs finish in milliseconds.
    viz_enabled=False skips matplotlib calls that would block in CI.
    """

    def _cfg(self, tmp_dir: str, epochs: int = 1) -> object:
        cfg_dict = {
            "data": {"input_dim": 16, "dataset": "mnist", "root": ".", "is_image": False},
            "model": {"model_type": "vae", "latent_dim": 4},
            "training": {
                "lr": 1e-2,
                "epochs": epochs,
                "batch_size": 4,
                "kl_weight": 1.0,
                "kl_warmup_epochs": 1,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "checkpoint_dir": tmp_dir,
                "validate_every_n_epochs": 1,
                "viz_every_n_epochs": 999,  # skip snapshots
                "early_stopping": False,
                "ema_decay": 0.999,
                "use_ema": False,
                "grad_clip": 0.0,
            },
            "wandb": {"enabled": False},
            "seed": 0,
            "experiment_name": "test",
        }
        return OmegaConf.create(cfg_dict)

    def test_returns_list_of_dicts(self, device, tmp_path):
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        loaders = _make_loaders(input_dim=16)
        cfg = self._cfg(str(tmp_path))

        history = train_model(model, loaders, cfg, device, run=_NoOpRun(), viz_enabled=False)

        assert isinstance(history, list)
        assert len(history) == 1
        assert isinstance(history[0], dict)

    def test_history_has_total_key(self, device, tmp_path):
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        loaders = _make_loaders(input_dim=16)
        cfg = self._cfg(str(tmp_path))

        history = train_model(model, loaders, cfg, device, run=_NoOpRun(), viz_enabled=False)

        assert "total" in history[0]

    def test_history_length_matches_epochs(self, device, tmp_path):
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        loaders = _make_loaders(input_dim=16)
        cfg = self._cfg(str(tmp_path), epochs=3)

        history = train_model(model, loaders, cfg, device, run=_NoOpRun(), viz_enabled=False)

        assert len(history) == 3

    def test_all_loss_values_are_positive(self, device, tmp_path):
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        loaders = _make_loaders(input_dim=16)
        cfg = self._cfg(str(tmp_path), epochs=2)

        history = train_model(model, loaders, cfg, device, run=_NoOpRun(), viz_enabled=False)

        for epoch_dict in history:
            assert epoch_dict["total"] > 0

    def test_model_parameters_change_after_training(self, device, tmp_path):
        """Optimizer must actually update model weights."""
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        initial_params = [p.data.clone() for p in model.parameters()]
        loaders = _make_loaders(input_dim=16)
        cfg = self._cfg(str(tmp_path), epochs=2)

        train_model(model, loaders, cfg, device, run=_NoOpRun(), viz_enabled=False)

        changed = any(
            not torch.allclose(p.data, p0)
            for p, p0 in zip(model.parameters(), initial_params, strict=False)
        )
        assert changed, "Model weights did not change — optimizer may not be working"

    def test_checkpoint_saved_at_first_epoch(self, device, tmp_path):
        """viz_every_n_epochs=1 triggers _save_checkpoint on epoch 1."""
        cfg_dict = {
            "data": {"input_dim": 16, "dataset": "mnist", "root": ".", "is_image": False},
            "model": {"model_type": "vae", "latent_dim": 4},
            "training": {
                "lr": 1e-2,
                "epochs": 1,
                "batch_size": 4,
                "kl_weight": 1.0,
                "kl_warmup_epochs": 1,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "checkpoint_dir": str(tmp_path),
                "validate_every_n_epochs": 1,
                "viz_every_n_epochs": 1,  # triggers on epoch 1
                "early_stopping": False,
                "ema_decay": 0.999,
                "use_ema": False,
                "grad_clip": 0.0,
            },
            "wandb": {"enabled": False},
            "seed": 0,
            "experiment_name": "test",
        }
        cfg = OmegaConf.create(cfg_dict)
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        loaders = _make_loaders(input_dim=16)

        train_model(model, loaders, cfg, device, run=_NoOpRun(), viz_enabled=False)

        assert (tmp_path / "epoch_001.pth").exists()

    def test_wandb_run_log_is_called(self, device, tmp_path):
        """run.log() must be called at least once per epoch."""
        model = _make_vae(input_dim=16, latent_dim=4).to(device)
        loaders = _make_loaders(input_dim=16)
        cfg = self._cfg(str(tmp_path), epochs=2)

        mock_run = mock.MagicMock()
        train_model(model, loaders, cfg, device, run=mock_run, viz_enabled=False)

        assert mock_run.log.call_count >= 2

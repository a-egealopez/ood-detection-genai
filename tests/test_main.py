"""
Tests for main entry point (main.py).

Comprehensive tests covering:
  - Argument parsing and validation
  - Short name building for experiment tracking
  - Checkpoint loading
  - Mode dispatch and basic execution flow
  - Command-line interface integration
"""

# Import main module functions
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    _load_checkpoint,
    _mode_distance_method,
    _mode_reconstruction_method,
    _mode_stats_summary,
    _parse_args,
    main,
)


class TestParseArgs:
    """Test argument parsing functionality."""

    def test_parse_args_reconstruction_method_defaults(self):
        """
        Happy path: Parse reconstruction-method with defaults.
        """
        with mock.patch("sys.argv", ["main.py", "--mode", "reconstruction-method"]):
            args = _parse_args()

        assert args.mode == "reconstruction-method"
        assert args.dataset == "mnist"
        assert args.experiment == "vae"
        assert args.skip_train is False
        assert args.skip_eval is False

    def test_parse_args_distance_method_defaults(self):
        """
        Happy path: Parse distance-method with defaults.
        """
        with mock.patch("sys.argv", ["main.py", "--mode", "distance-method"]):
            args = _parse_args()

        assert args.mode == "distance-method"
        assert args.distance_type == "knn"
        assert args.out == ""

    def test_parse_args_stats_summary_defaults(self):
        """
        Happy path: Parse stats-summary with defaults.
        """
        with mock.patch("sys.argv", ["main.py", "--mode", "stats-summary"]):
            args = _parse_args()

        assert args.mode == "stats-summary"
        assert args.logs_dir == "results/logs"
        assert args.out_csv == "results/summary/comparison.csv"

    def test_parse_args_custom_values(self):
        """
        Happy path: Parse with custom values.
        """
        with mock.patch(
            "sys.argv",
            [
                "main.py",
                "--mode",
                "reconstruction-method",
                "--dataset",
                "sicap",
                "--experiment",
                "ddpm",
                "--lr",
                "0.001",
                "--epochs",
                "100",
                "--seed",
                "42",
                "--skip-train",
                "--skip-eval",
            ],
        ):
            args = _parse_args()

        assert args.dataset == "sicap"
        assert args.experiment == "ddpm"
        assert args.lr == 0.001
        assert args.epochs == 100
        assert args.seed == 42
        assert args.skip_train is True
        assert args.skip_eval is True

    def test_parse_args_validation_distance_method_skip_flags(self):
        """
        Error case: distance-method with incompatible skip flags.
        """
        with mock.patch("sys.argv", ["main.py", "--mode", "distance-method", "--skip-train"]):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_parse_args_validation_stats_summary_extra_flags(self):
        """
        Error case: stats-summary with incompatible flags.
        """
        with mock.patch("sys.argv", ["main.py", "--mode", "stats-summary", "--skip-train"]):
            with pytest.raises(SystemExit):
                _parse_args()


class TestLoadCheckpoint:
    """Test checkpoint loading functionality."""

    def test_load_checkpoint_success(self, device):
        """
        Happy path: Load checkpoint successfully.
        """
        # Create mock checkpoint data
        model_state = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        checkpoint = {"model_state": model_state, "epoch": 10, "optimizer_state": {}}

        # Create mock model
        model = mock.MagicMock()
        model.load_state_dict = mock.MagicMock()

        with mock.patch("main.torch.load", return_value=checkpoint) as mock_load:
            with mock.patch("main.Path") as mock_path:
                mock_file = mock.MagicMock()
                mock_file.exists.return_value = True
                mock_path.return_value.__truediv__.return_value = mock_file

                # Load checkpoint
                _load_checkpoint(model, "test_experiment", device)

        # Verify torch.load was called
        mock_load.assert_called_once()
        # Verify model.load_state_dict was called with correct data
        model.load_state_dict.assert_called_once_with(model_state)

    def test_load_checkpoint_file_not_found(self, device):
        """
        Error case: Checkpoint file does not exist.
        """
        model = mock.MagicMock()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            _load_checkpoint(model, "nonexistent", device)

    def test_load_checkpoint_missing_model_state(self, device):
        """
        Error case: Checkpoint missing model_state key.
        """
        # Create checkpoint without model_state
        checkpoint = {"epoch": 10}

        model = mock.MagicMock()

        with mock.patch("main.Path") as mock_path_class:
            mock_ckpt_path = mock.MagicMock()
            mock_ckpt_path.exists.return_value = True
            mock_path_class.return_value.__truediv__.return_value = mock_ckpt_path

            with mock.patch("main.torch.load", return_value=checkpoint):
                with pytest.raises(KeyError, match="does not contain 'model_state'"):
                    _load_checkpoint(model, "test_experiment", device)


class TestModeStatsSummary:
    """Test stats-summary mode."""

    @mock.patch("main.run_summary")
    def test_mode_stats_summary(self, mock_run_summary):
        """
        Happy path: Execute stats-summary mode.
        """
        mock_run_summary.return_value = Path("results/summary/test.csv")

        args = mock.MagicMock()
        args.logs_dir = "results/logs"
        args.out_csv = "results/summary/test.csv"

        with mock.patch("builtins.print"):
            _mode_stats_summary(args)

        mock_run_summary.assert_called_once_with(
            logs_dir="results/logs", out_csv="results/summary/test.csv"
        )


class TestModeDistanceMethod:
    """Test distance-method mode."""

    @mock.patch("main.build_config")
    @mock.patch("main.seed_everything")
    @mock.patch("main.run_fm_distance")
    @mock.patch("artifacts.build_short_name")
    def test_mode_distance_method(self, mock_build_name, mock_run_fm, mock_seed, mock_build_cfg):
        """
        Happy path: Execute distance-method mode.
        """
        # Setup mocks
        cfg = mock.MagicMock()
        cfg.seed = 42
        cfg.experiment_name = "test_name"
        mock_build_cfg.return_value = cfg
        mock_build_name.return_value = "test_name"
        mock_run_fm.return_value = Path("results/test.json")

        args = mock.MagicMock()
        args.config = "configs/test.yaml"
        args.seed = 42
        args.distance_type = "knn"
        args.out = "results/test.json"

        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("builtins.print"):
                _mode_distance_method(args)

        # Verify calls
        mock_build_cfg.assert_called_once_with("configs/test.yaml")
        assert cfg.method == "distance-method"
        assert cfg.distance_type == "knn"
        assert cfg.experiment_name == "test_name"
        mock_seed.assert_called_once_with(42)
        mock_run_fm.assert_called_once_with(
            cfg=cfg, device=torch.device("cpu"), out="results/test.json"
        )


class TestModeReconstructionMethod:
    """Test reconstruction-method mode."""

    @mock.patch("main.build_config")
    @mock.patch("main.build_wandb_run")
    @mock.patch("main.build_dataloaders")
    @mock.patch("main.MODEL_REGISTRY")
    @mock.patch("main.stg")
    @mock.patch("main.seed_everything")
    @mock.patch("artifacts.build_short_name")
    @mock.patch("main._load_checkpoint")
    def test_mode_reconstruction_method_full_training(
        self,
        mock_load_ckpt,
        mock_build_name,
        mock_seed,
        mock_stg,
        mock_model_reg,
        mock_build_loaders,
        mock_build_wandb,
        mock_build_cfg,
    ):
        """
        Happy path: Execute reconstruction-method with full training.
        """
        # Setup mocks
        cfg = mock.MagicMock()
        cfg.method = "reconstruction-method"
        cfg.model.model_type = "vae"
        cfg.experiment_name = "test_exp"
        cfg.wandb.enabled = False
        cfg.seed = 42
        mock_build_cfg.return_value = cfg
        mock_build_name.return_value = "test_exp"

        run = mock.MagicMock()
        mock_build_wandb.return_value = run

        loaders = mock.MagicMock()
        mock_build_loaders.return_value = loaders

        model = mock.MagicMock()
        mock_model_reg.__getitem__.return_value = mock.MagicMock(return_value=model)

        results = {"auroc": 0.85}
        mock_stg.stage_evaluation.return_value = results

        args = mock.MagicMock()
        args.config = "configs/test.yaml"
        args.skip_train = False
        args.skip_eval = False

        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("builtins.print"):
                _mode_reconstruction_method(args)

        # Verify training flow
        mock_stg.stage_input_space_viz.assert_called_once_with(loaders, cfg, run)
        mock_stg.stage_training.assert_called_once_with(
            model, loaders, cfg, torch.device("cpu"), run
        )
        mock_stg.stage_evaluation.assert_called_once_with(
            model, loaders, cfg, torch.device("cpu"), run
        )
        mock_stg.print_summary.assert_called_once_with(cfg, results)

    @mock.patch("main.build_config")
    @mock.patch("main.build_wandb_run")
    @mock.patch("main.build_dataloaders")
    @mock.patch("main.MODEL_REGISTRY")
    @mock.patch("main.stg")
    @mock.patch("artifacts.build_short_name")
    @mock.patch("main._load_checkpoint")
    def test_mode_reconstruction_method_skip_train(
        self,
        mock_load_ckpt,
        mock_build_name,
        mock_stg,
        mock_model_reg,
        mock_build_loaders,
        mock_build_wandb,
        mock_build_cfg,
    ):
        """
        Happy path: Execute reconstruction-method skipping training (load checkpoint).
        """
        # Setup mocks
        cfg = mock.MagicMock()
        cfg.wandb.enabled = False
        cfg.method = "reconstruction-method"
        cfg.model.model_type = "vae"
        cfg.data.dataset = "mnist"
        cfg.seed = 42
        cfg.training.lr = 0.001
        cfg.training.epochs = 50
        mock_build_cfg.return_value = cfg
        mock_build_name.return_value = "recon_vae_mnist_s42_lr0.001_ep50"

        model = mock.MagicMock()
        mock_model_reg.__getitem__.return_value = mock.MagicMock(return_value=model)

        args = mock.MagicMock()
        args.skip_train = True
        args.skip_eval = False

        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("builtins.print"):
                _mode_reconstruction_method(args)

        # Verify checkpoint loading with correct experiment name
        mock_load_ckpt.assert_called_once_with(
            model, "recon_vae_mnist_s42_lr0.001_ep50", torch.device("cpu")
        )


class TestMainIntegration:
    """Integration tests for main function."""

    @mock.patch("main._parse_args")
    @mock.patch("main.MODE_DISPATCH")
    def test_main_dispatches_to_correct_mode(self, mock_dispatch, mock_parse_args):
        """
        Happy path: Main dispatches to correct mode handler.
        """
        args = mock.MagicMock()
        args.mode = "stats-summary"
        mock_parse_args.return_value = args

        mode_handler = mock.MagicMock()
        mock_dispatch.__getitem__.return_value = mode_handler

        main()

        mock_dispatch.__getitem__.assert_called_once_with("stats-summary")
        mode_handler.assert_called_once_with(args)

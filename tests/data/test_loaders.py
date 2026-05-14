"""
Tests for data loaders (src/data/loaders.py).

Happy path tests focus on expected behavior:
  - Tensor dataset creation with normalization
  - DataLoader building and configuration
  - Dataset metadata handling
  - Batch iteration with correct shapes
"""

from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loaders import (
    DataLoaderSpec,
    _to_tensor_dataset,
    build_dataloaders,
)


class TestDataLoaderSpec:
    """Test DataLoaderSpec dataclass."""

    def test_metadata_creation(self):
        """
        Happy path: DataLoaderSpec stores dataset information correctly.
        """
        metadata = DataLoaderSpec(
            id_name="MNIST",
            ood_name="SVHN",
            input_dim=784,
            is_image=True,
        )

        assert metadata.id_name == "MNIST"
        assert metadata.ood_name == "SVHN"
        assert metadata.input_dim == 784
        assert metadata.is_image is True

    def test_metadata_with_different_values(self):
        """
        Happy path: DataLoaderSpec works with different dataset configurations.
        """
        metadata = DataLoaderSpec(
            id_name="sicap",
            ood_name="sicap_ood",
            input_dim=2048,
            is_image=False,
        )

        assert metadata.id_name == "sicap"
        assert metadata.ood_name == "sicap_ood"
        assert metadata.input_dim == 2048
        assert metadata.is_image is False


class TestToTensorDataset:
    """Test tensor dataset creation with normalization."""

    def test_to_tensor_dataset_shapes(self):
        """
        Happy path: _to_tensor_dataset converts numpy arrays to TensorDataset with correct shapes.
        """
        batch_size = 10
        input_dim = 100

        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        y = np.random.randint(0, 10, batch_size).astype(np.int64)
        mean = np.zeros(input_dim)
        std = np.ones(input_dim)

        dataset = _to_tensor_dataset(x, y, mean, std)

        assert isinstance(dataset, TensorDataset)
        assert len(dataset) == batch_size

    def test_to_tensor_dataset_normalization(self):
        """
        Happy path: _to_tensor_dataset normalizes data (x - mean) / std.
        """
        batch_size = 10
        input_dim = 100

        # Create data with known mean and std
        x = np.ones((batch_size, input_dim), dtype=np.float32) * 10.0
        y = np.random.randint(0, 10, batch_size).astype(np.int64)
        mean = np.ones(input_dim) * 10.0
        std = np.ones(input_dim) * 2.0

        dataset = _to_tensor_dataset(x, y, mean, std)

        # Get first sample
        x_tensor, y_tensor = dataset[0]

        # After normalization (10 - 10) / 2 = 0
        assert torch.allclose(x_tensor, torch.zeros(input_dim), atol=1e-6)

    def test_to_tensor_dataset_dtype_conversion(self):
        """
        Happy path: _to_tensor_dataset converts to float32 for x, int64 for y.
        """
        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.int32)
        mean = np.zeros(2)
        std = np.ones(2)

        dataset = _to_tensor_dataset(x, y, mean, std)
        x_tensor, y_tensor = dataset[0]

        assert x_tensor.dtype == torch.float32
        assert y_tensor.dtype == torch.int64

    def test_to_tensor_dataset_std_zero_handling(self):
        """
        Edge case: _to_tensor_dataset divides by std (may produce NaN if std=0).
        This tests that normalization works as implemented (even if imperfect).
        Note: This is expected behavior - filtering near-zero std should be done upstream.
        """
        x = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)
        mean = np.array([1.0, 2.0])
        std = np.array([0.0, 1e-7])  # Near-zero std

        dataset = _to_tensor_dataset(x, y, mean, std)
        x_tensor, _ = dataset[0]

        # Check: First value should be inf/nan (0/0), second should be finite
        # This tests that the function behaves consistently with near-zero std
        assert x_tensor[0].item() != x_tensor[0].item()  # NaN check (NaN != NaN)
        assert torch.isfinite(x_tensor[1])  # Second component should be finite


class TestBuildDataloadersIntegration:
    """Test build_dataloaders with mocked datasets."""

    def test_build_dataloaders_returns_dict(self, mock_config_vae, mock_loaders_dict):
        """
        Happy path: build_dataloaders returns dict with train/id_eval/ood_eval loaders.
        Uses mock configuration and mocked MNIST dataset.
        """
        cfg = mock_config_vae

        # Mock torchvision datasets to avoid downloads
        with mock.patch("torchvision.datasets.MNIST") as mock_mnist:
            with mock.patch("torchvision.datasets.SVHN") as mock_svhn:
                # Create fake datasets
                fake_mnist_data = [(torch.randn(1, 28, 28), i % 10) for i in range(100)]
                fake_svhn_data = [(torch.randn(3, 32, 32), i % 10) for i in range(100)]

                mock_mnist.return_value = fake_mnist_data
                mock_svhn.return_value = fake_svhn_data

                loaders = build_dataloaders(cfg)

                # Verify structure
                assert isinstance(loaders, dict)
                assert "train" in loaders
                assert "id_eval" in loaders
                assert "ood_eval" in loaders

                # Verify all are DataLoaders
                for split in ["train", "id_eval", "ood_eval"]:
                    assert isinstance(loaders[split], DataLoader)

    def test_build_dataloaders_invalid_dataset(self, mock_config_vae):
        """
        Edge case: Invalid dataset name raises ValueError.
        """
        cfg = mock_config_vae
        cfg.data.dataset = "nonexistent_dataset"

        with pytest.raises(ValueError, match="Unsupported dataset"):
            build_dataloaders(cfg)

    def test_build_dataloaders_batch_size(self, mock_config_vae):
        """
        Happy path: Dataloaders respect batch_size from config.
        """
        cfg = mock_config_vae
        batch_size = 16
        cfg.training.batch_size = batch_size

        with mock.patch("torchvision.datasets.MNIST"):
            with mock.patch("torchvision.datasets.SVHN"):
                fake_mnist = [(torch.randn(1, 28, 28), i % 10) for i in range(100)]
                with mock.patch("torchvision.datasets.MNIST", return_value=fake_mnist):
                    with mock.patch("torchvision.datasets.SVHN", return_value=fake_mnist):
                        loaders = build_dataloaders(cfg)

                        # Check batch size in train loader
                        train_loader = loaders["train"]
                        assert train_loader.batch_size == batch_size


class TestDataLoaderIteration:
    """Test iteration and batching from loaders."""

    def test_mock_dataloader_iteration(self, mock_dataloader):
        """
        Happy path: Mock dataloader iterates correctly and produces batches.
        """
        batch_count = 0
        for x_batch, y_batch in mock_dataloader:
            batch_count += 1

            # Verify batch format
            assert isinstance(x_batch, torch.Tensor)
            assert isinstance(y_batch, torch.Tensor)
            assert x_batch.shape[1:] == (1, 784)  # (B, 1, 784)
            assert y_batch.shape[0] == x_batch.shape[0]

        assert batch_count > 0

    def test_mock_dataloader_shuffle(self):
        """
        Happy path: DataLoader shuffles training data (when shuffle=True).
        """
        x = torch.randn(20, 1, 784)
        y = torch.arange(20)
        dataset = TensorDataset(x, y)

        # Non-shuffled loader
        loader_no_shuffle = DataLoader(dataset, batch_size=4, shuffle=False)

        # Shuffled loader with fixed seed for determinism
        g = torch.Generator()
        g.manual_seed(99)
        loader_shuffle = DataLoader(dataset, batch_size=4, shuffle=True, generator=g)

        # Get all indices from shuffled loader
        indices_shuffled = []
        for _, y_batch in loader_shuffle:
            indices_shuffled.extend(y_batch.tolist())

        # Check that it's not in original order
        assert indices_shuffled != list(range(20))


class TestConfigIntegration:
    """Test configuration updates through build_dataloaders."""

    def test_config_updated_with_metadata(self, mock_config_vae):
        """
        Happy path: build_dataloaders updates config with dataset metadata.
        """
        cfg = mock_config_vae

        with mock.patch("torchvision.datasets.MNIST"):
            with mock.patch("torchvision.datasets.SVHN"):
                fake_data = [(torch.randn(1, 28, 28), i % 10) for i in range(50)]
                with mock.patch("torchvision.datasets.MNIST", return_value=fake_data):
                    with mock.patch("torchvision.datasets.SVHN", return_value=fake_data):
                        loaders = build_dataloaders(cfg)

                        # Config should be updated with metadata
                        assert cfg.data.input_dim == 784
                        assert cfg.data.id_name == "MNIST"
                        assert cfg.data.ood_name == "SVHN"
                        assert cfg.data.is_image is True


class TestDataLoaderNumWorkers:
    """Test num_workers configuration."""

    def test_dataloader_num_workers_config(self, mock_config_vae):
        """
        Happy path: DataLoader respects num_workers from config.
        """
        cfg = mock_config_vae
        cfg.data.num_workers = 0  # Set to 0 for testing (avoid multiprocessing issues)

        with mock.patch("torchvision.datasets.MNIST"):
            with mock.patch("torchvision.datasets.SVHN"):
                fake_data = [(torch.randn(1, 28, 28), i % 10) for i in range(50)]
                with mock.patch("torchvision.datasets.MNIST", return_value=fake_data):
                    with mock.patch("torchvision.datasets.SVHN", return_value=fake_data):
                        loaders = build_dataloaders(cfg)

                        # Verify num_workers is set correctly
                        assert loaders["train"].num_workers == 0


class TestDataLoaderPinMemory:
    """Test pin_memory configuration."""

    def test_dataloader_pin_memory_config(self, mock_config_vae):
        """
        Happy path: DataLoader respects pin_memory from config.
        """
        cfg = mock_config_vae
        cfg.data.pin_memory = False

        with mock.patch("torchvision.datasets.MNIST"):
            with mock.patch("torchvision.datasets.SVHN"):
                fake_data = [(torch.randn(1, 28, 28), i % 10) for i in range(50)]
                with mock.patch("torchvision.datasets.MNIST", return_value=fake_data):
                    with mock.patch("torchvision.datasets.SVHN", return_value=fake_data):
                        loaders = build_dataloaders(cfg)

                        # Verify pin_memory is set correctly
                        assert loaders["train"].pin_memory is False

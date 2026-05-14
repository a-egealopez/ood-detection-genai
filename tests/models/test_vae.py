import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.vae import VAEModel, build_vae_model


class TestVAEModelInitialization:
    def test_init_valid_parameters(self, device, mock_config_vae):
        model = VAEModel(784, 16, 1.0).to(device)
        assert model.input_dim == 784
        assert model.latent_dim == 16
        assert model.kl_weight == 1.0
        assert isinstance(model, nn.Module)

    def test_init_invalid_input_dim_too_large(self):
        with pytest.raises(ValueError, match="input_dim <= 1024"):
            VAEModel(input_dim=2048, latent_dim=32, kl_weight=1.0)


class TestVAEModelForwardPass:
    def test_forward_valid_batch(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        x_recon, z_mu, z_logvar = model(x)
        assert x_recon.shape == x.shape
        assert z_mu.shape == (x.shape[0], 16)
        assert z_logvar.shape == (x.shape[0], 16)
        assert x_recon.device == device
        assert z_mu.device == device
        assert z_logvar.device == device

    def test_forward_output_dtypes(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        x_recon, z_mu, z_logvar = model(x)
        assert x_recon.dtype == torch.float32
        assert z_mu.dtype == torch.float32
        assert z_logvar.dtype == torch.float32

    def test_forward_batch_size_1(self, device):
        x = torch.randn(1, 1, 784).to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        x_recon, z_mu, z_logvar = model(x)
        assert x_recon.shape == (1, 1, 784)
        assert z_mu.shape == (1, 16)


class TestVAEModelLoss:
    def test_loss_valid_inputs(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        x_recon, z_mu, z_logvar = model(x)
        loss_dict = model.loss(x, x_recon, z_mu, z_logvar, kl_weight=1.0)
        assert isinstance(loss_dict, dict)
        assert {"total", "recon", "kl"} <= loss_dict.keys()
        for val in loss_dict.values():
            assert isinstance(val, torch.Tensor)
            assert val.dim() == 0
        assert loss_dict["total"].item() >= 0
        assert loss_dict["recon"].item() >= 0
        assert loss_dict["kl"].item() >= 0

    def test_loss_composition(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        kl_weight = 0.5
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=kl_weight).to(device)
        x_recon, z_mu, z_logvar = model(x)
        loss_dict = model.loss(x, x_recon, z_mu, z_logvar, kl_weight=kl_weight)
        expected_total = loss_dict["recon"] + kl_weight * loss_dict["kl"]
        assert torch.allclose(loss_dict["total"], expected_total, atol=1e-6)

    def test_loss_with_custom_kl_weight(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        x_recon, z_mu, z_logvar = model(x)
        loss_w1 = model.loss(x, x_recon, z_mu, z_logvar, kl_weight=1.0)
        loss_w05 = model.loss(x, x_recon, z_mu, z_logvar, kl_weight=0.5)
        assert loss_w05["total"] <= loss_w1["total"]


class TestVAEModelEncode:
    def test_encode_valid_batch(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        z_mu, z_logvar = model.encode(x)
        assert z_mu.shape == (x.shape[0], 16)
        assert z_logvar.shape == (x.shape[0], 16)
        assert z_mu.dtype == torch.float32
        assert z_logvar.dtype == torch.float32

    def test_encode_no_grad(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        with torch.no_grad():
            z_mu, z_logvar = model.encode(x)
        assert z_mu.shape == (x.shape[0], 16)
        assert z_logvar.shape == (x.shape[0], 16)
        assert not z_mu.requires_grad
        assert not z_logvar.requires_grad


class TestVAEModelDecode:
    def test_decode_valid_latent(self, device):
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        z = torch.randn(4, 16).to(device)
        x_recon = model.decode(z)
        assert x_recon.shape == (4, 1, 784)
        assert x_recon.dtype == torch.float32

    def test_decode_single_sample(self, device):
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        z = torch.randn(1, 16).to(device)
        x_recon = model.decode(z)
        assert x_recon.shape == (1, 1, 784)


class TestVAEModelOODScore:
    def test_ood_score_valid_batch(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        scores = model.ood_score(x, mode="recon")
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (x.shape[0],)
        assert scores.dtype in [np.float32, np.float64]
        assert np.all(scores >= 0)

    def test_ood_score_modes(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        scores = model.ood_score(x, mode="recon")
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (4,)

    def test_set_ood_reference(self, device, sample_batch):
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        reference = {"mean": np.zeros(16), "cov_inv": np.eye(16)}
        model.set_ood_reference(reference)
        assert model.ood_reference is not None
        assert model.ood_reference == reference


class TestBuildModelFunction:
    def test_build_vae_model_vae(self, device, mock_config_vae):
        model = build_vae_model(mock_config_vae, device)
        assert isinstance(model, VAEModel)
        assert model.input_dim == 784
        assert model.latent_dim == 16
        assert model.kl_weight == 1.0

    def test_build_vae_model_returns_to_device(self, device, mock_config_vae):
        model = build_vae_model(mock_config_vae, device)
        for param in model.parameters():
            assert param.device == device


class TestVAEModelIntegration:
    def test_forward_loss_backward_cycle(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x_recon, z_mu, z_logvar = model(x)
        loss_dict = model.loss(x, x_recon, z_mu, z_logvar, kl_weight=1.0)
        assert loss_dict["total"].item() >= 0
        loss_dict["total"].backward()
        has_gradients = any(
            p.requires_grad and p.grad is not None for p in model.parameters()
        )
        assert has_gradients
        optimizer.step()

    def test_encode_decode_cycle(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = VAEModel(input_dim=784, latent_dim=16, kl_weight=1.0).to(device)
        z_mu, _ = model.encode(x)
        x_recon = model.decode(z_mu)
        assert x_recon.shape == x.shape
        assert torch.all(x_recon >= -2)
        assert torch.all(x_recon <= 2)

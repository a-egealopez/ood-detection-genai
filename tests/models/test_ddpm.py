import numpy as np
import torch
import torch.nn as nn

from src.models.ddpm import (
    DDPMModel,
    GroupNorm1d,
    ResidualMLPBlock,
    ResidualMLPDenoiser,
    SinusoidalPosEmb,
)


class TestSinusoidalPosEmb:
    def test_sinusoidal_embedding_shape(self, device):
        dim = 32
        emb = SinusoidalPosEmb(dim)
        t = torch.tensor([0, 50, 100, 200], device=device)
        output = emb(t)
        assert output.shape == (4, dim)
        assert output.dtype == torch.float32

    def test_sinusoidal_embedding_periodicity(self, device):
        dim = 32
        emb = SinusoidalPosEmb(dim)
        out1 = emb(torch.tensor([0], device=device))
        out2 = emb(torch.tensor([1], device=device))
        assert not torch.allclose(out1, out2)


class TestGroupNorm1d:
    def test_group_norm_shape(self, device):
        channels = 64
        norm = GroupNorm1d(channels, num_groups=8)
        x = torch.randn(4, channels, device=device)
        output = norm(x)
        assert output.shape == x.shape
        assert output.dtype == torch.float32

    def test_group_norm_normalization(self, device):
        channels = 64
        norm = GroupNorm1d(channels, num_groups=8)
        x = torch.randn(4, channels, device=device) * 100.0
        output = norm(x)
        assert output.std().item() < x.std().item()


class TestResidualMLPBlock:
    def test_residual_mlp_block_shape(self, device):
        hidden_dim = 128
        time_emb_dim = 32
        block = ResidualMLPBlock(hidden_dim, time_emb_dim, dropout=0.1)
        h = torch.randn(4, hidden_dim, device=device)
        t_emb = torch.randn(4, time_emb_dim, device=device)
        output = block(h, t_emb)
        assert output.shape == h.shape

    def test_residual_mlp_block_residual_connection(self, device):
        hidden_dim = 128
        time_emb_dim = 32
        block = ResidualMLPBlock(hidden_dim, time_emb_dim)
        h = torch.randn(4, hidden_dim, device=device, requires_grad=True)
        t_emb = torch.randn(4, time_emb_dim, device=device)
        output = block(h, t_emb)
        output.sum().backward()
        assert h.grad is not None


class TestResidualMLPDenoiser:
    def test_denoiser_forward_shape(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        denoiser = ResidualMLPDenoiser(
            input_dim=784, hidden_dim=128, depth=3, time_emb_dim=32
        ).to(device)
        t = torch.tensor([100, 200], device=device, dtype=torch.long)
        output = denoiser(x, t)
        assert output.shape == x.shape


def _make_ddpm(device, **kwargs) -> DDPMModel:
    defaults = dict(
        input_dim=784,
        hidden_dim=128,
        depth=3,
        time_emb_dim=32,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        prediction_type="epsilon",
        n_score_steps=10,
    )
    defaults.update(kwargs)
    return DDPMModel(**defaults).to(device)


class TestDDPMModelInitialization:
    def test_ddpm_init_valid_parameters(self, device, mock_config_ddpm):
        model = _make_ddpm(device, dropout=0.1)
        assert model.num_train_timesteps == 1000
        assert model.prediction_type == "epsilon"
        assert model.n_score_steps == 10
        assert isinstance(model, nn.Module)

    def test_ddpm_has_scheduler(self, device):
        model = _make_ddpm(device)
        assert hasattr(model, "scheduler")
        assert model.scheduler is not None


class TestDDPMModelForwardPass:
    def test_forward_valid_batch(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = _make_ddpm(device)
        x_recon, x_noisy, t = model(x)
        assert x_recon.shape == x.shape
        assert x_noisy.shape == x.shape
        assert t.shape == (x.shape[0],)
        assert t.dtype == torch.long

    def test_forward_output_devices(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = _make_ddpm(device)
        x_recon, x_noisy, t = model(x)
        assert x_recon.device == device
        assert x_noisy.device == device
        assert t.device == device


class TestDDPMModelLoss:
    def test_loss_epsilon_prediction(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = _make_ddpm(device)
        loss_dict = model.loss(x)
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert isinstance(loss_dict["total"], torch.Tensor)
        assert loss_dict["total"].dim() == 0
        assert loss_dict["total"].item() >= 0

    def test_loss_different_prediction_types(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        for pred_type in ["epsilon", "v_prediction", "sample"]:
            model = _make_ddpm(device, prediction_type=pred_type)
            assert model.loss(x)["total"].item() >= 0


class TestDDPMModelOODScore:
    def test_ood_score_valid_batch(self, device, sample_batch):
        x, _ = sample_batch
        x = x.to(device)
        model = _make_ddpm(device)
        scores = model.ood_score(x, mode="recon")
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (x.shape[0],)
        assert scores.dtype in [np.float32, np.float64]
        assert np.all(scores >= 0)

    def test_ood_score_modes(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device, n_score_steps=5)
        scores_recon = model.ood_score(x, mode="recon")
        assert scores_recon.shape == (x.shape[0],)


class TestDDPMModelReconstruction:
    def test_reconstruct_at_t(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device)
        t = torch.tensor([50, 100], device=device, dtype=torch.long)
        x_recon, x_noisy, pred = model.reconstruct_at_t(x, t)
        assert x_recon.shape == x.shape
        assert x_noisy.shape == x.shape
        assert pred.shape == x.shape

    def test_reconstruct_with_fixed_noise(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device)
        t = torch.tensor([50, 100], device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_recon1, _, _ = model.reconstruct_at_t(x, t, noise=noise)
        x_recon2, _, _ = model.reconstruct_at_t(x, t, noise=noise)
        assert torch.allclose(x_recon1, x_recon2, atol=1e-6)


class TestDDPMModelDenoisingTrajectory:
    def test_denoise_trajectory_structure(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device)
        states = model.denoise_trajectory(x, t_start=100, capture_timesteps=[100, 50, 25, 0])
        assert isinstance(states, dict)
        for t, x_t in states.items():
            assert isinstance(t, (int, torch.Tensor))
            assert isinstance(x_t, torch.Tensor)
            assert x_t.shape == x.shape

    def test_denoise_trajectory_produces_samples(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device)
        states = model.denoise_trajectory(x, t_start=50, capture_timesteps=[50, 25, 0])
        assert 0 in states
        assert states[0].shape == x.shape


class TestDDPMModelEncode:
    def test_encode_produces_latent(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device)
        z, logvar = model.encode(x, t_encode=40)
        assert z.shape[0] == x.shape[0]
        assert logvar is None


class TestDDPMModelODDReference:
    def test_set_ood_reference(self, device):
        model = _make_ddpm(device)
        reference = {"mean": np.zeros(784), "cov_inv": np.eye(784)}
        model.set_ood_reference(reference)
        assert model.ood_reference == reference

    def test_set_ood_reference_none(self, device):
        model = _make_ddpm(device)
        model.set_ood_reference(None)
        assert model.ood_reference is None


class TestDDPMModelIntegration:
    def test_training_step_full_cycle(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        model = _make_ddpm(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_dict = model.loss(x)
        loss_dict["total"].backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
        optimizer.step()

    def test_eval_mode_no_grad(self, device, sample_batch_small):
        x, _ = sample_batch_small
        x = x.to(device)
        x.requires_grad = True
        model = _make_ddpm(device, n_score_steps=5)
        model.eval()
        scores = model.ood_score(x, mode="recon")
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (x.shape[0],)

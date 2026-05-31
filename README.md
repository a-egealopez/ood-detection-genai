# OOD Detection with Generative Models

<p>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg">
  <img src="https://img.shields.io/badge/license-Apache%202.0-yellowgreen.svg">
</p>

Framework for **out-of-distribution (OOD) detection** using generative models. Compares reconstruction-based methods (VAE, DDPM) with distance-based baselines (KNN, Mahalanobis) across multiple datasets and scoring strategies.

OOD detection asks whether a test sample comes from the same distribution the model was trained on. Here, generative models are trained exclusively on in-distribution data and used at inference time to score samples — without any OOD examples during training. The primary application is histopathology: detecting high-grade prostate cancer (Gleason 3–4) as OOD with respect to a model trained on benign tissue (Gleason 1–2).

![DDPM denoising trajectory on MNIST](figures/results-evaluation-ddpm_mnist_s42_lr0.0001_t10-plots-ddpm_denoising_trajectory_compact.png)

Developed as a Bachelor's thesis — Universidad de Granada, 2026.

---

## Requirements

```bash
conda env create -f environment.yml
conda activate tfg
```

Python 3.10 · PyTorch 2.0+ · see `environment.yml` for the full dependency list.

---

## Project structure

```
.
├── configs/
│   ├── base.yaml                  # Shared defaults
│   ├── data/                      # Per-dataset configs (mnist, sicap_c1, sicap_c12, …)
│   └── experiments/               # Per-model configs (vae, ddpm, *_toy, *_path)
├── scripts/
│   ├── core.sh                    # Shared helpers and loop primitives
│   ├── experiments.sh             # Numbered experiment functions (exp_0 … exp_5)
│   └── run.sh                     # Entry point for multi-GPU runs
├── src/
│   ├── models/                    # VAE, DDPM, OOD scorers, base interface
│   ├── data/                      # Dataset loaders
│   ├── training/                  # Training loop, EMA, early stopping
│   ├── evaluation/                # Metrics, score extraction, plots
│   ├── benchmarking/              # Summary tables and ablation plots
│   ├── artifacts.py               # W&B logging and file persistence
│   ├── config.py                  # Config building and seeding
│   ├── stages.py                  # Pipeline stages (train, eval, visualise)
│   └── viz/                       # Matplotlib style helpers
└── main.py                        # CLI entry point
```

---

## Usage

The CLI exposes three modes via `--mode`.

### Reconstruction-based methods (VAE / DDPM)

```bash
# Train + evaluate
python main.py \
  --mode reconstruction-method \
  --experiment vae \
  --dataset mnist \
  --lr 1e-4 \
  --seed 42

# Evaluate only (requires a saved checkpoint)
python main.py \
  --mode reconstruction-method \
  --experiment vae \
  --dataset mnist \
  --skip-train
```

Supported experiments: `vae`, `ddpm`, `vae_toy`, `ddpm_toy`, `vae_path`, `ddpm_path`.  
Supported datasets: `mnist`, `sicap_c1`, `sicap_c12`, `pathmnist`, `moons`, `blobs`.

### Distance-based baselines

```bash
python main.py \
  --mode distance-method \
  --dataset sicap_c1 \
  --distance-type knn      # or: mahalanobis
```

No training required. Uses the training split as the reference distribution.

### Aggregate results

```bash
python main.py \
  --mode stats-summary \
  --logs-dir results/evaluation \
  --out-csv  results/summary/comparison.csv
```

Reads all `eval_results.json` files under `--logs-dir`, writes a CSV, and generates LaTeX tables and ablation plots under `results/summary/`.

---

## Running experiments

The numbered experiment functions in `scripts/experiments.sh` reproduce the full study.

```bash
# Run a single experiment group (e.g. toy datasets)
bash scripts/run.sh 1

# Run multiple groups on a specific GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh 1 2

# Typical 2-GPU split used in the thesis
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run.sh 1 2 > gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run.sh 3 4 5 > gpu1.log 2>&1 &

# Distance baselines (CPU, lightweight)
bash scripts/run.sh 0

# Generate summary after all runs finish
SUMMARIZE=1 bash scripts/run.sh
```

| Group | Content | Purpose |
|-------|---------|---------|
| `exp_0` | KNN + Mahalanobis baselines on SICAP | Distance-based reference, no training required |
| `exp_1` | VAE + DDPM on toy datasets (moons, blobs) | Sanity check on controlled 2D data |
| `exp_2` | VAE + DDPM on MNIST | Benchmark on standard image data (MNIST vs SVHN) |
| `exp_3` | VAE + DDPM on PathMNIST | Benchmark on histopathology images (normal vs tumour) |
| `exp_4` | VAE + DDPM on SICAP (multi-seed, multi-LR) | Main evaluation, results averaged across seeds |
| `exp_5` | DDPM scoring-step ablation (T ∈ {5, 10, 25, 50}) | Sensitivity of noise-based scores to number of steps |

Override defaults with environment variables:

```bash
SEEDS="42 107 2024" LRS_VAE="1e-4 5e-4" MODEL=vae bash scripts/run.sh 4
```

---

## OOD scoring modes

### VAE

| Mode | Description |
|------|-------------|
| `recon` | Per-sample MSE reconstruction error |
| `elbo` | Reconstruction + β · KL divergence |
| `latent_knn` | Distance to k-th nearest neighbour in latent space |
| `latent_mah` | Mahalanobis distance in latent space |

### DDPM

| Mode | Description |
|------|-------------|
| `noise_single` | MSE between predicted and actual noise at a fixed timestep |
| `noise_multi_mse` | Z-scored MSE aggregated over multiple timesteps |
| `noise_multi_cosine` | Z-scored cosine distance aggregated over multiple timesteps |
| `residual_mah` | Mahalanobis distance in the noise-residual space |
| `residual_knn` | KNN distance in the noise-residual space |

---

## Configuration

All hyperparameters live in YAML files. Experiment configs inherit from `configs/base.yaml` and override what they need.

```bash
# Start from the closest existing config
cp configs/experiments/vae.yaml configs/experiments/my_run.yaml
# Edit experiment_name and any hyperparameters, then:
python main.py --mode reconstruction-method --experiment my_run --dataset mnist
```

Results are written to `results/training/<experiment_id>/` and `results/evaluation/<experiment_id>/`.

---

## W&B integration

W&B logging is enabled by default. To disable:

```yaml
# configs/base.yaml or your experiment config
wandb:
  enabled: false
```
---

## Notes

This repository reflects the academic release of the thesis codebase.
It does not include a test suite.

---

## License

[Apache License 2.0](LICENSE) — © 2026 Alejandro Egea López / Universidad de Granada.
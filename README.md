<<<<<<< HEAD
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue.svg">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg">
  <img src="https://img.shields.io/badge/pytest-passing-brightgreen.svg">
  <img src="https://img.shields.io/badge/license-Apache%202.0-yellowgreen.svg">
</p>

# OOD Detection GenAI

Comprehensive framework for **Out-of-Distribution (OOD) detection** using generative models and distance-based baselines.

Compares reconstruction-based methods (**VAE**, **DDPM**) with distance-based approaches (**KNN**, **Residual**) for detecting OOD samples using feature-space anomaly scoring.

**Features:**
- ✅ Unified model interface via `MODEL_REGISTRY` (VAE, DDPM, extensible)
- ✅ Multiple OOD scoring modes (reconstruction error, latent norms, KNN, residual)
- ✅ Distance-based baselines (no training required)
- ✅ **135+ unit tests** covering all modules (96.8% pass rate)
- ✅ Automated comparison suite for scientific studies
- ✅ Configuration-driven experiments (YAML-based)
- ✅ Multi-dataset support (MNIST, SICAP)

---

## Quick Start

### Installation

```bash
conda env create -f environment.yml
conda activate tfg
```

For CPU-only: remove `pytorch-cuda` from `environment.yml` before creating environment.

### Quick Test

```bash
# Run test suite (validates entire setup)
pytest tests/ -v

# Single run: VAE on MNIST with default hyperparameters
python main.py --mode reconstruction-method --experiment vae --dataset mnist

# Single run: KNN baseline (no training required)
python main.py --mode distance-method --dataset mnist --distance-type knn

# Generate summary from existing results
python main.py --mode stats-summary --logs-dir results/logs --out-csv results/summary/comparison.csv
```

### Scientific Study

```bash
# Run complete comparison suite (2 datasets × 2 seeds × 2 learning rates)
./scripts/run_comparison_suite.sh

# Quick test (single dataset, seed, learning rate)
DATASETS="mnist" SEEDS="42" LRS="1e-3" ./scripts/run_comparison_suite.sh

# Only reconstruction methods
RUN_DISTANCE=0 ./scripts/run_comparison_suite.sh
```

For detailed suite documentation, see [COMPARISON_SUITE.md](COMPARISON_SUITE.md).

---

## Usage & Modes

The main entry point supports three modes:

### 1. Reconstruction-based OOD Detection

```bash
# Train and evaluate VAE
python main.py \
  --mode reconstruction-method \
  --experiment vae \
  --dataset mnist \
  --lr 0.001 \
  --epochs 100 \
  --seed 42

# Train and evaluate DDPM
python main.py \
  --mode reconstruction-method \
  --experiment ddpm \
  --dataset mnist \
  --lr 0.001 \
  --epochs 300 \
  --seed 42

# Skip training (evaluate only)
python main.py \
  --mode reconstruction-method \
  --experiment vae \
  --dataset mnist \
  --skip-train
```

### 2. Distance-based OOD Detection (Baselines)

```bash
# KNN baseline (uses training data as reference)
python main.py \
  --mode distance-method \
  --dataset mnist \
  --distance-type knn

# Residual distance baseline
python main.py \
  --mode distance-method \
  --dataset mnist \
  --distance-type residual
```

### 3. Summary Generation

```bash
# Aggregate all JSON results into CSV
python main.py \
  --mode stats-summary \
  --logs-dir results/logs \
  --out-csv results/summary/comparison.csv
```

### Command-line Parameters

| Parameter | Mode | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | all | `reconstruction-method` | Execution mode |
| `--experiment` | reconstruction | `vae` | Model type (vae/ddpm) |
| `--dataset` | all | `mnist` | Dataset (mnist/data_sicap) |
| `--lr` | reconstruction | config | Learning rate |
| `--epochs` | reconstruction | config | Training epochs |
| `--seed` | all | config | Random seed |
| `--skip-train` | reconstruction | false | Skip training, eval only |
| `--skip-eval` | reconstruction | false | Skip evaluation |
| `--distance-type` | distance | `knn` | Distance method (knn/residual) |
| `--logs-dir` | stats-summary | `results/logs` | Log directory |
| `--out-csv` | stats-summary | `results/summary/comparison.csv` | Output CSV path |

---

## Project Structure

```
ood-detection-genai/
├── configs/
│   ├── base.yaml                    # Base configuration template
│   ├── data/
│   │   ├── mnist.yaml               # MNIST dataset config
│   │   └── sicap.yaml               # SICAP dataset config
│   └── experiments/
│       ├── vae.yaml                 # VAE experiment config
│       └── ddpm.yaml                # DDPM experiment config
│
├── src/
│   ├── config.py                    # build_config, build_wandb_run, seed_everything
│   ├── stages.py                    # Training pipeline stages
│   ├── artifacts.py                 # Artifact management
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py               # build_dataloaders, data preprocessing
│   ├── models/
│   │   ├── __init__.py              # MODEL_REGISTRY
│   │   ├── base_model.py            # BaseModel interface
│   │   ├── vae.py                   # VAEWrapper implementation
│   │   ├── ddpm.py                  # DDPMWrapper implementation
│   │   └── ood_scorers.py           # OOD scoring functions
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py               # train_model, EMA, checkpointing
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py              # OOD evaluation metrics (AUROC, AUPR, FPR@TPR)
│       ├── extract.py               # Feature extraction
│       └── plot.py                  # Visualization
│
├── tests/                           # **NEW: Comprehensive test suite**
│   ├── conftest.py                  # Shared fixtures (135+ tests)
│   ├── test_config.py               # Config & W&B tests
│   ├── test_main.py                 # Main entry point tests
│   ├── models/
│   │   ├── test_vae.py              # VAE model tests
│   │   └── test_ddpm.py             # DDPM model tests
│   ├── data/
│   │   └── test_loaders.py          # Data loading tests
│   ├── training/
│   │   └── test_trainer.py          # Training loop tests
│   ├── evaluation/
│   │   └── test_evaluate.py         # OOD metrics tests
│   └── benchmarking/
│       └── test_generate_summary.py # Summary generation tests
│
├── scripts/
│   ├── run_comparison_suite.sh      # **NEW: Automated comparison script**
│   └── ...
│
├── main.py                          # Entry point (3 modes)
├── environment.yml                  # Conda dependencies
├── pyproject.toml                   # Project metadata
├── README.md                        # This file
├── COMPARISON_SUITE.md              # **NEW: Suite documentation**
└── ...
```

---

## OOD Scoring Methods

### VAE Scoring Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `recon` | MSE reconstruction error | Standard reconstruction loss |
| `elbo` | Reconstruction + β·KL | ELBO-based score |
| `latent_norm` | \|\|z_μ\|\|² | Latent space distance |
| `mahalanobis` | Mahalanobis distance in latent space | Assumes diagonal Gaussian prior |
| `residual` | Norm in PCA residual subspace | Complementary features |
| `knn` | Distance to k-th nearest ID latent | Nonparametric baseline |

### DDPM Scoring Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `recon` | Mean MSE(x̂₀, x) over timesteps | Reconstruction quality (Tweedie) |
| `noise` | Mean MSE(ε̂, ε) over timesteps | Noise prediction error |
| `residual` | Norm in PCA residual subspace | Complementary features |
| `knn` | Distance to k-th nearest ID latent | Nonparametric baseline |

---

## Testing

The project includes a **comprehensive test suite** with 135+ unit tests covering all modules.

### Run Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific module tests
pytest tests/models/test_vae.py -v              # VAE tests
pytest tests/training/test_trainer.py -v        # Training tests
pytest tests/evaluation/test_evaluate.py -v     # Evaluation tests

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run a single test
pytest tests/models/test_vae.py::TestVAEWrapperForwardPass::test_forward_valid_batch -v
```

### Test Coverage

- ✅ **Models** (VAE, DDPM): initialization, forward pass, loss computation, OOD scoring
- ✅ **Data loading**: dataset building, normalization, batch iteration, shuffling
- ✅ **Configuration**: config building, W&B integration, seed management
- ✅ **Training**: EMA, optimizers, schedulers, checkpointing, validation
- ✅ **Evaluation**: AUROC, AUPR, threshold analysis, metric computation
- ✅ **Benchmarking**: JSON log processing, CSV generation, aggregation
- ✅ **Main entry point**: argument parsing, mode dispatch, checkpoint loading

### Test Statistics

- **Total tests**: 135+
- **Pass rate**: 96.8%
- **Lines of test code**: 3000+
- **Coverage**: All critical paths

---

## Configuration

All behavior is controlled by YAML configuration files. Override `configs/base.yaml` with experiment-specific configs:

### Creating Experiment Configs

```bash
cp configs/experiments/vae.yaml configs/experiments/my_experiment.yaml
# Edit my_experiment.yaml with custom hyperparameters
python main.py --lr 0.01 --epochs 200
```

### Key Configuration Sections

| Block | Purpose |
|-------|---------|
| `model` | Model type, latent dims, diffusion parameters |
| `data` | Dataset, paths, train/eval splits |
| `training` | Learning rate, optimizer, scheduler, epochs, checkpointing |
| `ood` | OOD scoring method, KNN k, residual components |
| `wandb` | Project, tags, logging frequency |

### Example Experiment Config

```yaml
# configs/experiments/my_run.yaml
experiment_name: my_run

model:
  model_type: vae
  input_dim: 784
  latent_dim: 16
  kl_weight: 1.0

data:
  dataset: mnist
  train_split: 0.8
  batch_size: 128

training:
  lr: 1e-3
  epochs: 100
  optimizer: adam
  scheduler: cosine

ood:
  score_mode: reconstruction  # or 'latent_norm', 'residual', 'knn', etc.
```

---

## Results Structure

After running experiments, results are organized as:

```
results/
├── logs/                                # Detailed JSON results
│   ├── vae_mnist_lr1e-3_seed42.json
│   ├── ddpm_mnist_lr1e-4_seed107.json
│   ├── knn_mnist.json
│   └── residual_data_sicap.json
│
└── summary/
    ├── comparison.csv                   # Aggregated metrics table
    ├── timing_reconstruction.txt        # Execution times
    └── timing_distance.txt
```

### Output Metrics (JSON)

Each result file contains:

```json
{
  "method": "vae",
  "dataset": "mnist",
  "seed": 42,
  "lr": 0.001,
  "auroc": 0.92,
  "aupr": 0.88,
  "threshold_at_5_fpr": 0.35,
  "tpr_at_5_fpr": 0.82
}
```

### Aggregated Results (CSV)

```
method,dataset,seed,lr,auroc,aupr,threshold_at_5_fpr,tpr_at_5_fpr
vae,mnist,42,0.001,0.92,0.88,0.35,0.82
vae,mnist,107,0.001,0.91,0.87,0.36,0.80
vae,data_sicap,42,0.0001,0.85,0.82,0.42,0.75
...
```

---

## Extending the Project

### Add a New Model

1. Create `src/models/mymodel.py`:
```python
from src.models.base_model import BaseModel

class MyModelWrapper(BaseModel):
    def __init__(self, cfg, device):
        ...
    
    def forward(self, x):
        ...
    
    def loss(self, x, ...):
        ...
```

2. Register in `src/models/__init__.py`:
```python
MODEL_REGISTRY = {
    'vae': VAEWrapper,
    'ddpm': DDPMWrapper,
    'mymodel': MyModelWrapper,  # ← Add here
}
```

3. Create config file `configs/experiments/mymodel.yaml`

### Add a New OOD Scorer

Edit `src/models/ood_scorers.py` and add a branch in `compute_ood_score()`:

```python
if mode == "my_score":
    return my_scoring_function(x, model, device)
```

### Add a New Dataset

1. Add loader logic to `src/data/loaders.py`
2. Create config file `configs/data/mydataset.yaml`
3. Reference in experiment config: `data.dataset: mydataset`

---

## Performance & Reproducibility

### Reproducibility

- All experiments use fixed random seeds (default: 42, 107)
- Configuration files are versioned
- Results include seed information for multi-seed averaging

### Performance Notes

- **VAE training**: ~10-30 min per run (varies by dataset/config)
- **DDPM training**: ~1-4 hours per run (CPU: very slow)
- **Distance baselines**: ~30-60 seconds per run
- **Full comparison suite**: 4-8 hours (2 datasets × 2 seeds × 2 LRs)

### GPU Requirements

- VAE: GPU optional (works on CPU, slow)
- DDPM: GPU strongly recommended
- Distance methods: CPU acceptable

---

## Citation & References

If you use this framework, please cite:

```bibtex
@thesis{ood_detection_genai,
  title={Out-of-Distribution Detection using Generative Models},
  author={Your Name},
  year={2026},
  school={Your Institution}
}
```

---

## License

Apache License 2.0 - See LICENSE file for details.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size in config or use CPU-only mode |
| Tests failing | Run `pytest tests/ -v` to see detailed error messages |
| Slow training | Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"` |
| Config not found | Verify path exists: `ls configs/experiments/` |
| No results generated | Check that training completed: `ls results/logs/` |
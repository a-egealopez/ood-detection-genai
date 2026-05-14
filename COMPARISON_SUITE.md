# OOD Detection GenAI - Comparison Suite

## Overview

The `run_comparison_suite.sh` script provides an automated benchmarking suite for comparing different OOD detection methods:

- **Reconstruction-based methods**: VAE, DDPM
- **Distance-based methods**: KNN, Residual features

Recommended for scientific studies with systematic hyperparameter exploration.

## Quick Start

```bash
# Run complete study (default: 2 datasets, 2 seeds, 2 learning rates)
./scripts/run_comparison_suite.sh

# Quick test (single dataset, seed, learning rate)
DATASETS="mnist" SEEDS="42" LRS="1e-3" ./scripts/run_comparison_suite.sh

# Only reconstruction or distance methods
RUN_DISTANCE=0 ./scripts/run_comparison_suite.sh

# Skip training (eval only with pre-trained models)
SKIP_TRAIN=1 ./scripts/run_comparison_suite.sh
```

## Configuration

Environment variables to customize the suite:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASETS` | `mnist data_sicap` | Space-separated dataset names |
| `SEEDS` | `42 107` | Space-separated random seeds (reproducibility) |
| `LRS` | `1e-3 1e-4` | Space-separated learning rates |
| `MODELS` | `vae ddpm` | Space-separated generative models to test |
| `RUN_RECONSTRUCTION` | `1` | Enable reconstruction methods (1/0) |
| `RUN_DISTANCE` | `1` | Enable distance methods (1/0) |
| `SKIP_TRAIN` | `0` | Skip training, eval only (1/0) |

## Complete Study Configuration

```bash
# Scientific study: 2 models × 2 datasets × 2 seeds × 2 learning rates = 16 runs
./scripts/run_comparison_suite.sh

# Extended study: Add more learning rates
LRS="1e-3 1e-4 5e-4" ./scripts/run_comparison_suite.sh

# Extended study: Include seed 123 as well
SEEDS="42 107 123" ./scripts/run_comparison_suite.sh
```

## Output Structure

Results saved in `results/` directory:

```
results/
├── logs/                           # Detailed JSON results
│   ├── vae_mnist_lr1e-3_seed42.json
│   ├── vae_mnist_lr1e-3_seed107.json
│   ├── vae_data_sicap_lr1e-3_seed42.json
│   ├── ddpm_mnist_lr1e-4_seed42.json
│   ├── knn_mnist.json
│   ├── residual_mnist.json
│   └── ...
└── summary/
    ├── comparison.csv              # Aggregated metrics table
    ├── timing_reconstruction.txt   # Runtime stats (reconstruction)
    └── timing_distance.txt         # Runtime stats (distance)
```

## Output Metrics

Each JSON result file contains:

```json
{
  "method": "vae",                   # Method name
  "dataset": "mnist",                # Dataset used
  "seed": 42,                        # Random seed
  "lr": 0.001,                       # Learning rate (if applicable)
  "auroc": 0.92,                     # Area under ROC curve
  "aupr": 0.88,                      # Area under PR curve  
  "thresholds": {
    "fpr_5": {"tpr": 0.85, "threshold": 0.3},
    "fpr_10": {"tpr": 0.92, "threshold": 0.2}
  }
}
```

## Workflow

### 1. Reconstruction Methods
- Trains VAE/DDPM on training data
- Evaluates OOD detection on validation/OOD splits  
- Saves results with hyperparameters
- **Time per run**: ~5-10 min (VAE), ~30-60 min (DDPM)

### 2. Distance Methods
- Computes KNN and Residual distances
- Uses training data for feature reference (no training needed)
- Saves results immediately
- **Time per run**: ~30-60 seconds

### 3. Summary Generation
- Aggregates all JSON files → CSV
- Computes mean/std across seeds
- Enables comparison across hyperparameters

## Statistical Analysis After Suite

Once the suite completes, analyze results:

```bash
# View aggregated results
cat results/summary/comparison.csv

# Check execution times
cat results/summary/timing_*.txt

# Custom analysis in Python
python3 -c "
import pandas as pd
df = pd.read_csv('results/summary/comparison.csv')
print(df.groupby(['method', 'dataset']).agg({'auroc': ['mean', 'std']}))
"
```

## Performance & Time Estimates

**Default configuration (2 datasets × 2 seeds × 2 LRs):**

| Phase | Runs | Est. Time |
|-------|------|-----------|
| VAE reconstruction | 8 | ~40-80 min |
| DDPM reconstruction | 8 | ~240-480 min |
| KNN distance | 2 | ~2 min |
| Residual distance | 2 | ~2 min |
| **Total** | **20** | **~4-8 hours** |

**Tips to reduce time:**
- Run only one model: `MODELS="vae"` (~2-4 hours)
- Skip DDPM training: `MODELS="vae"` + distance methods
- Run in background: `nohup ./scripts/run_comparison_suite.sh > suite.log 2>&1 &`

## Integration with Testing

Validate setup before running suite:

```bash
# Run unit tests
pytest tests/ -v

# Quick sanity check (1 seed, 1 dataset, 1 LR)
DATASETS="mnist" SEEDS="42" LRS="1e-3" ./scripts/run_comparison_suite.sh

# Full study after validation
./scripts/run_comparison_suite.sh
```

## Reproducibility Notes

- All runs use fixed random seeds (42, 107) for reproducibility
- Results are aggregated per seed for statistical analysis
- Store this script version with results for reference:
  ```bash
  cp scripts/run_comparison_suite.sh results/suite_config_$(date +%Y%m%d).sh
  ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Config not found" | Ensure `configs/experiments/` has vae.yaml, ddpm.yaml |
| Slow execution | DDPM training is slow by default; use `MODELS="vae"` for testing |
| Partial results | Check `results/logs/` - may be incomplete runs; restart with `SKIP_TRAIN=1` |
| Memory errors | Reduce batch size in config files or use fewer seeds |
| GPU out of memory | Run on CPU-only: `CUDA_VISIBLE_DEVICES="" ./scripts/run_comparison_suite.sh` |


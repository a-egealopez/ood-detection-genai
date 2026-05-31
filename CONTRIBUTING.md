# Contributing

Internal conventions for a clean and reproducible commit history.

---

## Commit messages

Format: `<type>: <short description>`

| Type | When to use |
|------|-------------|
| `feat` | New feature or model |
| `fix` | Bug fix |
| `exp` | New experiment or config variant |
| `refactor` | Code reorganisation with no behaviour change |
| `docs` | Documentation changes |
| `results` | Plots, metrics, or experiment outputs |
| `chore` | Maintenance (dependencies, cleanup) |

Examples:

```
feat: add latent centroid OOD score to VAE evaluation
fix: normalize features using train statistics in test split
exp: ddpm reconstruction sweep t* = [100, 250, 500, 750]
results: add ROC curves for VAE vs DDPM comparison
refactor: simplify score mode dispatch in scoring
```

---

## Running a new experiment

### 1. Create a config

Copy the closest existing experiment config and edit it:

```bash
cp configs/experiments/vae.yaml configs/experiments/my_experiment.yaml
```

Set a unique `experiment_name` — this controls where all outputs are written:

```
results/training/<experiment_name>/
results/evaluation/<experiment_name>/
```

### 2. Run

```bash
python main.py \
  --mode reconstruction-method \
  --experiment my_experiment \
  --dataset sicap_c1 \
  --lr 1e-4 \
  --seed 42
```

Use `--skip-train` to re-run evaluation on an existing checkpoint without retraining, and `--skip-eval` to train only.

For DDPM, the number of scoring steps can be overridden at eval time without retraining:

```bash
python main.py \
  --mode reconstruction-method \
  --experiment ddpm \
  --dataset sicap_c1 \
  --n-score-steps 25 \
  --skip-train
```

### 3. Inspect outputs

Plots land under `results/evaluation/<experiment_name>/plots/`. Metrics are written to `results/evaluation/<experiment_name>/eval_results.json`.

To aggregate results across multiple runs into a single CSV:

```bash
python main.py --mode stats-summary \
  --logs-dir results/evaluation \
  --out-csv  results/summary/comparison.csv
```

### 4. Commit

Commit the `.yaml` config together with any results you want to preserve, using the `exp:` prefix:

```
exp: vae sicap_c1 latent_dim=64 lr=1e-4
results: ROC curves and eval_results for vae sicap_c1 run
```

This keeps a clear link between a config and the outputs it produced.

---

## Code style

The project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. Configuration lives in `pyproject.toml`.

### Setup

Pre-commit hooks run Ruff automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

After that, `ruff check` and `ruff format` run on every `git commit`. Hooks are defined in `.pre-commit-config.yaml`.

### Running manually

```bash
# Check for lint errors
ruff check .

# Auto-fix what can be fixed
ruff check --fix .

# Format code (equivalent to black)
ruff format .

# Check formatting without modifying files
ruff format --check .
```

CI enforces both on every push and pull request to `main` (see `.github/workflows/lint.yml`). A commit that fails either check will be blocked.
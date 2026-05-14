# Development Guide

Internal conventions to keep a clean and reproducible commit history
throughout the project.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable code only. Merged into when something works and is clean. |
| `dev` | Day-to-day development branch. |
| `exp/<name>` | Isolated experiment branch, e.g. `exp/ddpm-t500-ablation` |

Typical workflow:

```bash
# Day-to-day work happens on dev
git checkout dev

# When an experiment requires significant changes, create its own branch
git checkout -b exp/ddpm-t500-ablation

# Once it works and is clean, merge back to dev
git checkout dev
git merge exp/ddpm-t500-ablation

# When dev is stable, merge to main
git checkout main
git merge dev
```

---

## Commit Messages

Format: `<type>: <short description>`

| Type | When to use |
|------|-------------|
| `feat` | New feature or model |
| `fix` | Bug fix |
| `exp` | New experiment or config variant |
| `refactor` | Code reorganisation with no behaviour change |
| `docs` | Changes to README, CONTRIBUTING, or other documentation |
| `results` | Adding plots, metrics, or experiment results |
| `chore` | Maintenance tasks (dependencies, cleanup) |

Examples:

```
feat: add latent centroid OOD score to VAE evaluation
fix: normalize features using train statistics in test split
exp: ddpm reconstruction sweep t* = [100, 250, 500, 750]
results: add ROC curves for VAE vs DDPM comparison
docs: update README with configuration system section
refactor: simplify score mode dispatch in scoring
```

---

## Running a New Experiment

1. Copy the closest existing config:
   ```bash
   cp configs/experiments/vae.yaml configs/experiments/my_experiment.yaml
   ```
2. Update `experiment_name` so results are saved in their own subfolder.
3. Run and inspect plots under `results/<experiment_name>/plots/`.
4. Commit the `.yaml` file with the `exp:` prefix for full traceability between results and configurations.
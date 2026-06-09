import argparse
from pathlib import Path

import torch

from src import stages as stg
from src.artifacts import build_experiment_id, build_wandb_run
from src.benchmarking import (
    run_aggregate_tables,
    run_feature_distance,
    run_summary,
    run_t_ablation,
    run_training_times,
)
from src.config import build_config, seed_everything
from src.data import build_dataloaders
from src.models import MODEL_REGISTRY
from src.stages import StageContext

_TOY_DATASETS = frozenset({"moons", "blobs"})

RECON_VISUALIZERS = {
    "vae": stg.stage_reconstruction_viz_vae,
    "ddpm": stg.stage_reconstruction_viz_ddpm,
    "toy": stg.stage_reconstruction_viz_toy,
}

MODE_LABELS = {
    "reconstruction-method": "Reconstruction-based",
    "distance-method": "Distance-based",
    "stats-summary": "Stats Summary",
}

SCORE_LABELS: dict[str, str] = {
    "recon": "Reconstruction Error",
    "elbo": "ELBO",
    "latent_knn": "Latent k-NN",
    "latent_mah": "Latent Mahalanobis",
    "noise_single": "Single-step MSE",
    "noise_multi_mse": "Multi-step MSE (z-score)",
    "noise_multi_cosine": "Multi-step Cosine (z-score)",
    "recon_single": "Single-step Reconstruction",
    "recon_multi": "Multi-step Reconstruction",
    "residual_mah": "Residual Mahalanobis",
    "residual_knn": "Residual k-NN",
}


def _parse_args():
    p = argparse.ArgumentParser(description="Generative model pipeline.")

    p.add_argument(
        "--mode",
        choices=["reconstruction-method", "distance-method", "stats-summary"],
        default="reconstruction-method",
    )
    p.add_argument(
        "--dataset",
        choices=["mnist", "sicap_c1", "sicap_c12", "moons", "blobs", "pathmnist", "pathmnist_c1", "pathmnist_c2"],
        default="mnist",
    )
    p.add_argument("--seed", type=int)
    p.add_argument("--config", type=str, default="configs/base.yaml")

    p.add_argument("--experiment", type=str, default="mlp/vae/sicap")
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")

    p.add_argument(
        "--n-score-steps",
        type=int,
        default=None,
        help="Override DDPM n_score_steps for noise_multi ablation (eval only, no retrain).",
    )
    p.add_argument("--distance-type", choices=["knn", "mahalanobis"], default="knn")
    p.add_argument("--out", default="")

    p.add_argument("--logs-dir", default="results/evaluation")
    p.add_argument("--out-csv", default="results/summary/comparison.csv")

    args = p.parse_args()

    if args.mode == "distance-method" and (args.skip_train or args.skip_eval):
        p.error("--skip-train/--skip-eval not applicable in distance-method mode")
    if args.mode == "stats-summary" and (args.skip_train or args.skip_eval or args.out != ""):
        p.error("--skip-train/--skip-eval/--out not applicable in stats-summary mode")
    if args.n_score_steps is not None and args.mode != "reconstruction-method":
        p.error("--n-score-steps is only valid with --mode reconstruction-method")

    return args


def _mode_stats_summary(args):
    from src.viz.style import apply_paper_style

    apply_paper_style(None)

    out_csv = run_summary(results_dir=args.logs_dir, out_csv=args.out_csv, labels_map=SCORE_LABELS)
    print(f"saved: {out_csv}")
    out_dir = str(Path(out_csv).parent)
    run_aggregate_tables(csv_path=out_csv, out_dir=out_dir)
    run_t_ablation(csv_path=out_csv, out_dir=out_dir)
    run_training_times(out_dir=out_dir)


def _mode_distance_method(args):
    cfg = build_config(args.experiment, args.dataset)

    from src.viz.style import apply_paper_style

    apply_paper_style(cfg)

    if args.seed is not None:
        cfg.seed = args.seed

    cfg.method = "distance-method"
    cfg.distance_type = args.distance_type

    short_name = build_experiment_id(cfg)
    cfg.experiment_name = short_name
    cfg.evaluation = cfg.get("evaluation", {})
    cfg.evaluation.results_dir = f"results/evaluation/{short_name}"
    print(f"{MODE_LABELS['distance-method']} [{args.distance_type}] — {short_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    out_path = run_feature_distance(cfg=cfg, device=device, out=args.out)
    print(f"saved: {out_path}")


def _mode_reconstruction_method(args):
    cfg = build_config(args.experiment, args.dataset)
    cfg.evaluation = cfg.get("evaluation", {})

    from src.viz.style import apply_paper_style

    apply_paper_style(cfg)

    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed

    cfg.method = "reconstruction-method"
    train_experiment_id = build_experiment_id(cfg)
    cfg.training.checkpoint_dir = f"results/training/{train_experiment_id}/checkpoints"
    cfg.viz.train_plots_dir = f"results/training/{train_experiment_id}/plots"

    if args.n_score_steps is not None and "ddpm" in str(cfg.model.get("model_type", "")):
        cfg.model.n_score_steps = args.n_score_steps

    if "ddpm" in str(cfg.model.get("model_type", "")):
        eval_experiment_id = f"{train_experiment_id}_t{int(cfg.model.get('n_score_steps', 10))}"
    else:
        eval_experiment_id = train_experiment_id
    cfg.experiment_name = eval_experiment_id
    cfg.viz.eval_plots_dir = f"results/evaluation/{eval_experiment_id}/plots"
    cfg.evaluation.results_dir = f"results/evaluation/{eval_experiment_id}"

    cfg.wandb.run_name = eval_experiment_id

    print(f"{MODE_LABELS[cfg.method]} — {cfg.experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    run = build_wandb_run(cfg)
    loaders = build_dataloaders(cfg)
    model = MODEL_REGISTRY[cfg.model.model_type](cfg, device)

    ctx = StageContext(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
        run=run,
        experiment_id=build_experiment_id(cfg),
    )

    try:
        if args.skip_train:
            ckpt_path = Path(cfg.training.checkpoint_dir) / "model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train first.")
            model.load(str(ckpt_path), device)
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            cfg.viz.plots_dir = str(cfg.viz.train_plots_dir)
            stg.stage_input_space_viz(ctx)
            torch.cuda.empty_cache()
            stg.stage_training(ctx)

        if not args.skip_eval:
            cfg.viz.plots_dir = str(cfg.viz.eval_plots_dir)
            mt = str(cfg.model.model_type)
            viz_key = "toy" if cfg.data.dataset in _TOY_DATASETS else ("vae" if "vae" in mt else "ddpm")
            if visualize := RECON_VISUALIZERS.get(viz_key):
                visualize(ctx)
            results = stg.stage_evaluation(ctx)
            stg.print_summary(cfg, results)
    finally:
        if cfg.wandb.enabled:
            run.finish()


MODE_DISPATCH = {
    "stats-summary": _mode_stats_summary,
    "distance-method": _mode_distance_method,
    "reconstruction-method": _mode_reconstruction_method,
}


def main():
    args = _parse_args()
    MODE_DISPATCH[args.mode](args)


if __name__ == "__main__":
    main()

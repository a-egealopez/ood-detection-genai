import argparse
from pathlib import Path

import torch

from src import stages as stg
from src.artifacts import build_experiment_id, build_wandb_run
from src.benchmarking import run_aggregate_tables, run_feature_distance, run_summary, run_t_ablation
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


def _parse_args():
    p = argparse.ArgumentParser(description="Generative model pipeline.")

    p.add_argument(
        "--mode",
        choices=["reconstruction-method", "distance-method", "stats-summary"],
        default="reconstruction-method",
    )
    p.add_argument("--dataset", choices=["mnist", "sicap_c1", "sicap_c12", "moons", "blobs", "pathmnist"], default="mnist")
    p.add_argument("--seed", type=int)
    p.add_argument("--config", type=str, default="configs/base.yaml")

    p.add_argument("--experiment", choices=["vae", "ddpm", "vae_toy", "ddpm_toy", "vae_path", "ddpm_path"], default="vae")
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")

    p.add_argument("--n-score-steps", type=int, default=None,
                   help="Override DDPM n_score_steps for noise_multi ablation (eval only, no retrain).")
    p.add_argument("--distance-type", choices=["knn", "mahalanobis"], default="knn")
    p.add_argument("--out", default="")

    p.add_argument("--logs-dir", default="results")
    p.add_argument("--out-csv", default="results/summary/comparison.csv")

    args = p.parse_args()

    if args.mode == "distance-method" and (args.skip_train or args.skip_eval):
        p.error("--skip-train/--skip-eval not applicable in distance-method mode")
    if args.mode == "stats-summary" and (args.skip_train or args.skip_eval or args.out != ""):
        p.error("--skip-train/--skip-eval/--out not applicable in stats-summary mode")
    if args.n_score_steps is not None and (
        args.mode != "reconstruction-method" or "ddpm" not in args.experiment
    ):
        p.error("--n-score-steps is only valid with --mode reconstruction-method and a ddpm experiment")

    return args


def _mode_stats_summary(args):
    out_csv = run_summary(results_dir=args.logs_dir, out_csv=args.out_csv)
    print(f"saved: {out_csv}")
    out_dir = str(Path(out_csv).parent)
    run_aggregate_tables(csv_path=out_csv, out_dir=out_dir)
    run_t_ablation(csv_path=out_csv, out_dir=out_dir)


def _mode_distance_method(args):
    cfg = build_config(args.experiment, args.dataset)

    if args.seed is not None:
        cfg.seed = args.seed

    cfg.method = "distance-method"
    cfg.distance_type = args.distance_type

    short_name = build_experiment_id(cfg)
    cfg.experiment_name = short_name
    print(f"{MODE_LABELS['distance-method']} [{args.distance_type}] — {short_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    out_path = run_feature_distance(cfg=cfg, device=device, out=args.out)
    print(f"saved: {out_path}")


def _mode_reconstruction_method(args):
    cfg = build_config(args.experiment, args.dataset)

    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed

    cfg.method = "reconstruction-method"

    
    cfg.experiment_name = build_experiment_id(cfg)


    if args.n_score_steps is not None and str(cfg.model.get("model_type", "")) == "ddpm":
        frozen_ckpt_dir = str(cfg.training.checkpoint_dir)  # resolves ${experiment_name} now
        cfg.model.n_score_steps = args.n_score_steps
        cfg.training.checkpoint_dir = frozen_ckpt_dir        # pin to training-time path
        cfg.experiment_name = build_experiment_id(cfg)       # now encodes the new T

    cfg.wandb.run_name = cfg.experiment_name

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
            stg.stage_input_space_viz(ctx)
            torch.cuda.empty_cache()
            stg.stage_training(ctx)

        if not args.skip_eval:
            viz_key = "toy" if cfg.data.dataset in _TOY_DATASETS else cfg.model.model_type
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

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from src.viz.style import apply_paper_style


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.no_improve = 0

    def step(self, val_loss: float) -> bool:
        if val_loss + self.min_delta < self.best_val:
            self.best_val = val_loss
            self.no_improve = 0
        else:
            self.no_improve += 1
        return self.no_improve >= self.patience


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> optim.Optimizer:
    opt_name = str(cfg.training.get("optimizer", "adam")).lower()
    lr = float(cfg.training.lr)
    wd = float(cfg.training.get("weight_decay", 0.0))
    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(cfg: DictConfig, optimizer: optim.Optimizer):
    sched_name = str(cfg.training.get("scheduler", "none")).lower()
    if sched_name != "cosine":
        return None
    epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    min_lr = float(cfg.training.get("min_lr", 1e-6))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def _validation_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    kl_weight: float,
    ema: "EMA | None" = None,
) -> float:
    if ema is not None:
        raw_state = {k: v.clone() for k, v in model.state_dict().items() if k in ema.shadow}
        ema.copy_to(model)

    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            losses = model.compute_loss(x, kl_weight)
            values.append(float(next(iter(losses.values())).item()))
    model.train()

    if ema is not None:
        state = model.state_dict()
        for k, v in raw_state.items():
            state[k].copy_(v)

    return float(np.mean(values)) if values else float("inf")


def _training_snapshot(
    model: torch.nn.Module,
    loaders: dict,
    cfg: DictConfig,
    device: torch.device,
    epoch: int,
    epochs: int,
    run,
) -> None:
    fig = model.snapshot_fig(loaders, cfg, device, epoch, epochs)

    try:
        from src.artifacts import save_figure

        plots_dir = Path(cfg.viz.get("train_plots_dir", "results/training/plots/"))
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_figure(
            fig=fig,
            out_path=plots_dir / f"training_snapshot_epoch_{epoch:03d}.png",
            run=run if getattr(cfg, "wandb", {}).get("enabled", False) else None,
            image_key=f"train/snapshot/epoch_{epoch}",
            artifact_type="plot",
            artifact_prefix="snapshot",
            metadata={"epoch": epoch},
            png_dpi=int(cfg.viz.get("savefig_dpi", 300)),
        )
    finally:
        plt.close(fig)


def _print_training_header(model: torch.nn.Module, cfg: DictConfig) -> None:
    model_type = cfg.model.model_type
    t = cfg.training

    print(f"\n{'=' * 52}")
    print(f"  {model_type.upper()} training")
    print(f"{'=' * 52}")
    print(f"  epochs        : {t.epochs}")
    print(f"  lr            : {t.lr}")
    print(f"  optimizer     : {t.get('optimizer', 'adamw')}")
    print(f"  scheduler     : {t.get('scheduler', 'none')}")

    grad_clip = float(t.get("grad_clip", 0.0))
    if grad_clip > 0:
        print(f"  grad_clip     : {grad_clip}")

    if bool(t.get("use_ema", False)):
        print(f"  ema_decay     : {t.get('ema_decay', 0.999)}")

    if bool(t.get("early_stopping", False)):
        print(f"  early_stop    : patience={t.get('early_stopping_patience', 20)}")

    info = model.training_info()
    print(f"  {'—' * 46}")
    for k, v in info.items():
        label = f"{k:<14}"
        val = f"{v:,}" if isinstance(v, int) else v
        print(f"  {label}: {val}")
    print(f"{'=' * 52}\n")


_PRINT_SKIP_KEYS = {"recon"}


def _update_pbar(
    pbar,
    epoch_losses: dict,
    val_loss: float | None,
    lr: float,
    avg_gnorm: float | None,
    kl_weight: float,
) -> None:
    postfix = {k: f"{v:.4f}" for k, v in epoch_losses.items() if k not in _PRINT_SKIP_KEYS}
    if kl_weight > 0:
        postfix["kl_w"] = f"{kl_weight:.2e}"
    if val_loss is not None:
        postfix["val"] = f"{val_loss:.4f}"
    postfix["lr"] = f"{lr:.2e}"
    if avg_gnorm is not None:
        postfix["gnorm"] = f"{avg_gnorm:.3f}"
    pbar.set_postfix(postfix)
    pbar.close()


def _print_epoch_log(
    model_type: str,
    epoch: int,
    epochs: int,
    epoch_losses: dict,
    val_loss: float | None,
    lr: float,
    avg_gnorm: float | None,
    kl_weight: float,
) -> None:
    parts = [f"{k}={v:.4f}" for k, v in epoch_losses.items() if k not in _PRINT_SKIP_KEYS]
    if kl_weight > 0:
        parts.append(f"kl_w={kl_weight:.2e}")
    if val_loss is not None:
        parts.append(f"val={val_loss:.4f}")
    parts.append(f"lr={lr:.2e}")
    if avg_gnorm is not None:
        parts.append(f"gnorm={avg_gnorm:.3f}")
    print(f"[{model_type.upper()}] Epoch {epoch:03d}/{epochs}  " + "  ".join(parts), flush=True)


def train_model(
    model: torch.nn.Module,
    loaders: dict,
    cfg: DictConfig,
    device: torch.device,
    run=None,
    viz_enabled: bool = False,
) -> list[dict]:
    from src.artifacts import _NoOpRun

    run = run or _NoOpRun()
    model_type = cfg.model.model_type
    t_cfg = cfg.training

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    ema = (
        EMA(model, float(t_cfg.get("ema_decay", 0.999)))
        if bool(t_cfg.get("use_ema", False))
        else None
    )
    grad_clip = float(t_cfg.get("grad_clip", 0.0))
    val_every = int(t_cfg.get("validate_every_n_epochs", 1))
    viz_every = int(t_cfg.viz_every_n_epochs)
    epochs = int(t_cfg.epochs)

    stopper = (
        EarlyStopper(
            patience=int(t_cfg.get("early_stopping_patience", 20)),
            min_delta=float(t_cfg.get("early_stopping_min_delta", 0.0)),
        )
        if bool(t_cfg.get("early_stopping", False))
        else None
    )

    _print_training_header(model, cfg)
    try:
        apply_paper_style(cfg)
    except Exception:
        pass
    history: list[dict] = []
    _interactive = sys.stdin.isatty()

    for epoch in range(1, epochs + 1):
        kl_weight = model.kl_weight_at(epoch, cfg)

        model.train()
        totals: dict[str, float] = {}
        n = 0
        grad_norm_total = 0.0
        train_loader = loaders["train"]
        pbar = tqdm(
            total=len(train_loader),
            desc=f"[{model_type.upper()}] Epoch {epoch:03d}/{epochs}",
            unit="batch",
            leave=True,
            disable=not _interactive,
        )

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            L = model.compute_loss(x, kl_weight)
            next(iter(L.values())).backward()

            if grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norm_total += gn.item()
            optimizer.step()
            if ema is not None:
                ema.update(model)

            for k, v in L.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n += 1
            pbar.update(1)

            if _interactive:
                postfix = {k: f"{totals[k] / n:.4f}" for k in totals if k not in _PRINT_SKIP_KEYS}
                if kl_weight > 0:
                    postfix["kl_w"] = f"{kl_weight:.2e}"
                pbar.set_postfix(postfix)

        epoch_losses = {k: v / n for k, v in totals.items()}
        current_lr = optimizer.param_groups[0]["lr"]
        avg_gnorm = grad_norm_total / n if grad_clip > 0 else None
        history.append(epoch_losses)

        wandb_payload = {f"train/loss_{k}": v for k, v in epoch_losses.items()}
        wandb_payload["train/kl_weight"] = kl_weight
        wandb_payload["train/lr"] = current_lr
        run.log(wandb_payload, step=epoch)

        if scheduler is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The epoch parameter in", category=UserWarning
                )
                scheduler.step()

        val_loss = None
        if epoch % val_every == 0:
            val_loss = _validation_loss(model, loaders["id_eval"], device, kl_weight, ema=ema)
            run.log({"val/loss_total": val_loss}, step=epoch)

            if stopper is not None and stopper.step(val_loss):
                _update_pbar(pbar, epoch_losses, val_loss, current_lr, avg_gnorm, kl_weight)
                if not _interactive:
                    _print_epoch_log(model_type, epoch, epochs, epoch_losses, val_loss, current_lr, avg_gnorm, kl_weight)
                print(f"Early stopping at epoch {epoch} (best val={stopper.best_val:.6f})")
                break

        _update_pbar(pbar, epoch_losses, val_loss, current_lr, avg_gnorm, kl_weight)
        if not _interactive:
            _print_epoch_log(model_type, epoch, epochs, epoch_losses, val_loss, current_lr, avg_gnorm, kl_weight)

        if viz_enabled and (epoch % viz_every == 0 or epoch == 1):
            _training_snapshot(model, loaders, cfg, device, epoch, epochs, run)

    if ema is not None:
        ema.copy_to(model)

    ckpt_dir = Path(t_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / "model.pth"))
    print(f"{model_type.upper()} training complete. Checkpoint saved to {ckpt_dir / 'model.pth'}")
    return history

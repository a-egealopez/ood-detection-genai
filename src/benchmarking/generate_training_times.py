import re
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "moons": r"\textit{Moons}",
    "blobs": r"\textit{Blobs}",
    "mnist": "MNIST",
    "pathmnist_c1": "PathMNIST~(C1)",
    "pathmnist_c2": "PathMNIST~(C2)",
    "sicap_c1": "SICAP-C1",
    "sicap_c12": "SICAP-C12",
}
MODELS = {
    ("dist", "knn"): "KNN",
    ("dist", "mahalanobis"): "Mahalanobis",
    ("mlp", "vae"): "MLP-VAE",
    ("mlp", "ddpm"): "MLP-DDPM",
    ("unet", "vae"): "UNet-VAE",
    ("unet", "ddpm"): "UNet-DDPM",
}
GROUPS = [
    ("Sintéticos", ["moons", "blobs"]),
    ("Imágenes", ["mnist", "pathmnist_c1", "pathmnist_c2"]),
    ("Histología", ["sicap_c1", "sicap_c12"]),
]

_TIME_RE = re.compile(r"\[time\]\s+(\S+)\s+->\s+(\d+):(\d+):(\d+)")
_EPOCHS_RE = re.compile(r"epochs\s*:\s*(\d+)")
_EARLY_STOP_RE = re.compile(r"Early stopping at epoch (\d+)")


def _fmt(s):
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h}\\,h\\,{m:02d}\\,m"
    if m:
        return f"{m}\\,m\\,{s:02d}\\,s"
    return f"{s}\\,s"


def _fmt_cell(secs: int, epochs: int | None) -> str:
    t = _fmt(secs)
    if epochs is not None:
        return rf"{t}\,({epochs})"
    return t


def _parse(path):
    p = path.split("/")
    if len(p) >= 4:
        return (p[0], p[1]), p[3], "/".join(p[4:])
    if len(p) == 2 and p[0].startswith("dist-"):
        return ("dist", p[0][5:]), p[1], ""
    return None


def run_training_times(
    log_dir: Path | str | list[Path | str] = "logs",
    out_dir: Path | str = "results/summary",
    file_model_filter: dict[str, set] | None = None,
) -> Path:
    """Parse training-time logs and emit a LaTeX tabular with avg time and epochs.

    Distinguishes training runs from eval-only runs:
      - Training:  line matching ``epochs : N`` sets the training flag.
      - Eval-only: line containing ``Loaded checkpoint:`` clears the flag.
    Only [time] entries reached during a training run are recorded.
    Epoch count is the early-stopping epoch when triggered, else the
    configured max-epochs for that run.

    file_model_filter: restrict accepted model keys per file path
                       (e.g. to extract only VAE entries from a mixed file).
    """
    out_dir = Path(out_dir)

    if isinstance(log_dir, (list, tuple)):
        log_files: list[Path] = [Path(f) for f in log_dir]
    else:
        log_files = sorted(Path(log_dir).glob("log_*.txt"))

    last: dict = {}        # (model, dataset, seed_lr) → seconds
    last_epochs: dict = {} # (model, dataset, seed_lr) → epoch count

    for f in log_files:
        if not f.exists():
            continue
        allowed = file_model_filter.get(str(f)) if file_model_filter else None
        # is_eval_run: flipped to True only when "Loaded checkpoint:" appears
        # (indicating an eval-only run). KNN/MLP reruns without "epochs :" are
        # treated as training because they never emit "Loaded checkpoint:".
        is_eval_run = False
        current_max_epochs: int | None = None
        pending_stop_epoch: int | None = None

        for line in f.read_text(errors="replace").splitlines():
            m_ep = _EPOCHS_RE.search(line)
            if m_ep:
                current_max_epochs = int(m_ep.group(1))
                is_eval_run = False  # explicit training header
                continue

            if "Loaded checkpoint:" in line:
                is_eval_run = True   # eval-only run detected
                pending_stop_epoch = None
                continue

            m_es = _EARLY_STOP_RE.search(line)
            if m_es:
                pending_stop_epoch = int(m_es.group(1))
                continue

            m = _TIME_RE.search(line)
            if not m:
                continue

            # Snapshot and reset per-run state
            epoch = pending_stop_epoch if pending_stop_epoch is not None else current_max_epochs
            pending_stop_epoch = None
            skip = is_eval_run
            is_eval_run = False  # reset: next run unknown until a marker appears

            if skip:
                continue

            parsed = _parse(m.group(1))
            if parsed is None:
                continue
            model, dataset, seed_lr = parsed
            if allowed is not None and model not in allowed:
                continue
            if dataset not in DATASETS or model not in MODELS:
                continue

            secs = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            key = (model, dataset, seed_lr)
            last[key] = secs
            if epoch is not None:
                last_epochs[key] = epoch

    agg: dict = defaultdict(list)
    for (model, dataset, _), secs in last.items():
        agg[(model, dataset)].append(secs)
    data = {k: int(statistics.mean(v)) for k, v in agg.items()}

    agg_ep: dict = defaultdict(list)
    for (model, dataset, _), ep in last_epochs.items():
        agg_ep[(model, dataset)].append(ep)
    data_epochs = {k: int(round(statistics.mean(v))) for k, v in agg_ep.items()}

    models = [mk for mk in MODELS if any((mk, ds) in data for _, dss in GROUPS for ds in dss)]
    n_cols = 1 + len(models)

    lines = [
        r"\begin{tabular}{l" + "r" * len(models) + "}",
        r"    \toprule",
        "    Conjunto & " + " & ".join(MODELS[mk] for mk in models) + r" \\",
    ]

    for group_name, datasets in GROUPS:
        rows = []
        for ds in datasets:
            cells = [r"\quad " + DATASETS[ds]] + [
                _fmt_cell(data[(mk, ds)], data_epochs.get((mk, ds)))
                if (mk, ds) in data else "---"
                for mk in models
            ]
            if any(c != "---" for c in cells[1:]):
                rows.append("    " + " & ".join(cells) + r" \\")
        if not rows:
            continue
        lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{{n_cols}}}{{l}}{{\textbf{{{group_name}}}}} \\")
        lines.extend(rows)

    lines += [r"    \bottomrule", r"\end{tabular}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "table_training_times.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabla guardada en → {out}")
    return out

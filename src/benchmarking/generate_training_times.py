import re
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "moons": r"\textit{Moons}", "blobs": r"\textit{Blobs}",
    "mnist": "MNIST",
    "pathmnist_c1": "PathMNIST~(C1)", "pathmnist_c2": "PathMNIST~(C2)",
    "sicap_c1": "SICAP-C1", "sicap_c12": "SICAP-C12",
}
MODELS = {
    ("dist", "knn"):          "KNN",
    ("dist", "mahalanobis"):  "Mahalanobis",
    ("mlp",  "vae"):          "MLP-VAE",
    ("mlp",  "ddpm"):         "MLP-DDPM",
    ("unet", "vae"):          "UNet-VAE",
    ("unet", "ddpm"):         "UNet-DDPM",
}
GROUPS = [
    ("Sintéticos", ["moons", "blobs"]),
    ("Imágenes",   ["mnist", "pathmnist_c1", "pathmnist_c2"]),
    ("Histología", ["sicap_c1", "sicap_c12"]),
]

_TIME_RE = re.compile(r"\[time\]\s+(\S+)\s+->\s+(\d+):(\d+):(\d+)")


def _fmt(s):
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    if h: return f"{h}\\,h\\,{m:02d}\\,m"
    if m: return f"{m}\\,m\\,{s:02d}\\,s"
    return f"{s}\\,s"


def _parse(path):
    p = path.split("/")
    if len(p) >= 4:
        return (p[0], p[1]), p[3], "/".join(p[4:])
    if len(p) == 2 and p[0].startswith("dist-"):
        return ("dist", p[0][5:]), p[1], ""
    return None


def run_training_times(log_dir: Path | str = "logs", out_dir: Path | str = "results/summary") -> Path:
    log_dir = Path(log_dir)
    out_dir = Path(out_dir)

    last: dict = {}
    for f in sorted(log_dir.glob("log_*.txt")):
        for line in f.read_text(errors="replace").splitlines():
            m = _TIME_RE.search(line)
            if not m:
                continue
            parsed = _parse(m.group(1))
            if parsed is None:
                continue
            model, dataset, seed_lr = parsed
            if dataset not in DATASETS or model not in MODELS:
                continue
            secs = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
            last[(model, dataset, seed_lr)] = secs

    agg: dict = defaultdict(list)
    for (model, dataset, _), secs in last.items():
        agg[(model, dataset)].append(secs)
    data = {k: int(statistics.mean(v)) for k, v in agg.items()}

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
                _fmt(data[(mk, ds)]) if (mk, ds) in data else "---"
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
